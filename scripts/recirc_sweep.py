#!/usr/bin/env python
"""Training-free recirculation (Mozer et al., 2608.17981) on a Mote checkpoint — the paper's
source/destination sweep replicated on the main network. Eval-only; runs on the CPU so it can sit
beside a training arm.

Recirculation at (s, d, alpha): after chunk t's ordinary (first) pass through the main network, the
residual stream after block s is rescaled to the L2 norm of the stream after block d and mixed in,

    z'_d = beta * z_d + alpha * (|z_d| / |z_s|) * z_s        (beta = 1 - alpha unless --beta1)

then blocks d+1..L-1 are re-run from z'_d and their {P2, I~} cache rows for chunk t are overwritten.
Chunk t's own readout stays the first pass (the paper's figure 3c: recurrence in depth AND step); the
recirculated state reaches every later chunk through the cache. The encoder, router and chunking are
computed once per window and shared by every configuration — they run before the main network.

    PYTHONPATH=. .venv/bin/python scripts/recirc_sweep.py --ckpt runs/t3l_dense_4e-4/last.pt --data data/local_mix
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mote.config import MoteConfig  # noqa: E402
from mote.data.loader import ByteShard  # noqa: E402
from mote.model.dc import ste_ones  # noqa: E402
from mote.model.hnet import HNetForCausalLM  # noqa: E402
from mote.train.train import load_checkpoint  # noqa: E402

LN2 = math.log(2)


def load_model(path: Path, ema: bool):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MoteConfig.from_dict(ck["config"])
    model = HNetForCausalLM(cfg, device=torch.device("cpu"))
    load_checkpoint(path, model, ck=ck)
    if ema:
        vals = ck["extra"]["ema"]
        with torch.no_grad():
            for p, v in zip(model.parameters(), vals):
                p.copy_(v.to(p.dtype))
    model.eval()
    return model, cfg, int(ck["step"])


@torch.no_grad()
def front(model, inputs):
    """Everything before the main network: embeddings, encoder, router, chunking. Shared by all configs."""
    B, L = inputs.shape
    mask = torch.ones(B, L, dtype=torch.bool)
    h = model.embeddings(inputs)
    x = model.encoder(h)
    h, residual = x, model.residual_proj(x.float())
    routing = model.routing_module(h, mask)
    hc, _next_mask = model.chunk_layer(h, routing.boundary_mask)
    return h, residual, routing, model._pad(hc)


@torch.no_grad()
def back(model, zc, h, residual, routing, targets):
    """Everything after the main network: dechunk, decoder, head, summed NLL."""
    D0 = model.cfg.d_model_outer
    z = model.dechunk_layer(zc[..., :D0], routing.boundary_mask, routing.boundary_prob)
    y = z.float() * ste_ones(routing.selected_probs.float())
    h3 = model.decoder((y + residual).to(h.dtype))
    logits = model.head_logits(h3)
    V = logits.shape[-1]
    return float(F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1), reduction="sum"))


@torch.no_grad()
def main_sequential(model, x, s: int, d: int, alpha: float, beta1: bool = False, ramp: int = 0):
    """The main network chunk by chunk with a per-layer {P2, I~} cache, recirculating (s -> d) at
    `alpha`. alpha = 0 reproduces the ordinary forward (the harness's own correctness check)."""
    blocks = model.main_network.layers
    L = len(blocks)
    B, M, D = x.shape
    caches = [None] * L
    outs = []
    for t in range(M):
        hidden, residual = x[:, t : t + 1], None
        zs, new = [], []
        for l, blk in enumerate(blocks):
            hidden, residual, c = blk(hidden, residual, cache=caches[l], return_cache=True)
            new.append(c)
            zs.append(hidden.float() + residual.float())  # the residual stream after block l
        outs.append(model.main_network.rmsnorm(hidden, residual=residual, prenorm=False, residual_in_fp32=True))
        a = alpha * (min(t / ramp, 1.0) if ramp else 1.0)
        if a > 0:
            zd, zsrc = zs[d], zs[s]
            scale = zd.norm(dim=-1, keepdim=True) / zsrc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            b = 1.0 if beta1 else (1.0 - a)
            hidden, residual = (b * zd + a * scale * zsrc).to(x.dtype), None
            for l in range(d + 1, L):  # second pass: overwrite chunk t's rows at the layers above d
                hidden, residual, c = blocks[l](hidden, residual, cache=caches[l], return_cache=True)
                new[l] = c
        caches = new
    return torch.cat(outs, dim=1)


def windows(shard, n_windows: int, batch_size: int, seq_len: int):
    n_batches = max(1, n_windows // batch_size)
    for ids, _mask in shard.sequential_batches(batch_size, seq_len, n_batches, spread=True):
        yield ids[:, :-1], ids[:, 1:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/local_mix")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--alpha", type=float, default=0.10, help="mixture coefficient for the (s, d) grid")
    ap.add_argument("--pairs", default="all", help="'all' = every s > d, or 's:d,s:d,...' (block indices, 0-based stream after the block)")
    ap.add_argument("--alphas", default="0.04,0.07,0.15,0.25", help="alpha scan on the best grid pair ('' = skip)")
    ap.add_argument("--variants", action="store_true", help="also try beta=1 and a 10-chunk ramp on the best pair")
    ap.add_argument("--ema", action="store_true", help="evaluate the EMA weights instead of the raw ones")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=None, help="prefix for .json/.md (default docs/results/<date>-recirc-sweep)")
    ap.add_argument("--check-only", action="store_true", help="only the harness correctness check")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    model, cfg, step = load_model(Path(args.ckpt), args.ema)
    L = cfg.main.n_layers
    shard = ByteShard(args.data, "val")
    print(f"ckpt {args.ckpt} step {step} ({'ema' if args.ema else 'raw'}) — main {L} blocks, d_model {cfg.main.d_model}; "
          f"{args.windows} val windows x {args.seq_len} bytes, spread", flush=True)

    # front half once; keep everything the configs share
    t0 = time.time()
    cache = []
    n_tok, n_chunks = 0, 0
    for inputs, targets in windows(shard, args.windows, args.batch_size, args.seq_len):
        h, residual, routing, x = front(model, inputs)
        cache.append((h, residual, routing, x, targets))
        n_tok += targets.numel()
        n_chunks += int(routing.boundary_mask.sum())
    print(f"front half: {len(cache)} batches, {n_tok} targets, bpic {n_tok / max(n_chunks, 1):.3f}, {time.time() - t0:.1f} s", flush=True)

    # correctness: the parallel forward vs. the sequential cache path at alpha = 0
    t0 = time.time()
    nll_par = sum(back(model, model.main_network(x), h, r, rt, tg) for h, r, rt, x, tg in cache)
    nll_seq = sum(back(model, main_sequential(model, x, 1, 0, 0.0), h, r, rt, tg) for h, r, rt, x, tg in cache)
    base = nll_par / n_tok / LN2
    print(f"baseline val_bpb {base:.5f} (parallel) vs {nll_seq / n_tok / LN2:.5f} (sequential, alpha=0) — "
          f"{time.time() - t0:.1f} s for both", flush=True)
    if abs(nll_seq - nll_par) / n_tok / LN2 > 1e-3:
        print("FAIL: the sequential path does not reproduce the forward; fix the harness before reading any sweep", flush=True)
        return 1
    if args.check_only:
        return 0

    def run(s, d, alpha, beta1=False, ramp=0):
        t1 = time.time()
        nll = sum(back(model, main_sequential(model, x, s, d, alpha, beta1, ramp), h, r, rt, tg) for h, r, rt, x, tg in cache)
        bpb = nll / n_tok / LN2
        rec = {"s": s, "d": d, "alpha": alpha, "beta": 1.0 if beta1 else round(1 - alpha, 4), "ramp": ramp,
               "val_bpb": round(bpb, 5), "delta": round(bpb - base, 5), "pct": round(100 * (bpb - base) / base, 3),
               "seconds": round(time.time() - t1, 1)}
        print(f"  s={s} d={d} alpha={alpha:.2f} beta={rec['beta']} ramp={ramp}: {bpb:.5f} ({rec['pct']:+.2f} %) {rec['seconds']} s", flush=True)
        return rec

    if args.pairs == "all":
        pairs = [(s, d) for d in range(L - 1) for s in range(d + 1, L)]
    else:
        pairs = [tuple(int(v) for v in p.split(":")) for p in args.pairs.split(",")]
    print(f"grid: {len(pairs)} (s, d) pairs at alpha {args.alpha}", flush=True)
    results = [run(s, d, args.alpha) for s, d in pairs]
    best = min(results, key=lambda r: r["val_bpb"])
    print(f"best pair: s={best['s']} d={best['d']} {best['val_bpb']:.5f} ({best['pct']:+.2f} %)", flush=True)

    scan = []
    if args.alphas:
        print("alpha scan on the best pair", flush=True)
        scan = [run(best["s"], best["d"], float(a)) for a in args.alphas.split(",")]
    variants = []
    if args.variants:
        print("variants on the best pair", flush=True)
        variants = [run(best["s"], best["d"], args.alpha, beta1=True), run(best["s"], best["d"], args.alpha, ramp=10)]

    out = Path(args.out) if args.out else ROOT / "docs" / "results" / f"{time.strftime('%Y-%m-%d')}-recirc-sweep"
    everything = results + scan + variants
    overall = min(everything, key=lambda r: r["val_bpb"])
    payload = {"ckpt": args.ckpt, "step": step, "weights": "ema" if args.ema else "raw", "windows": args.windows,
               "seq_len": args.seq_len, "n_targets": n_tok, "bpic": round(n_tok / max(n_chunks, 1), 4),
               "baseline_val_bpb": round(base, 5), "grid_alpha": args.alpha, "grid": results, "alpha_scan": scan,
               "variants": variants, "best": overall}
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    # markdown: % change grid (rows = source s, cols = destination d)
    lines = [f"# Training-free recirculation sweep — {Path(args.ckpt).parent.name} (step {step}, {payload['weights']} weights)", "",
             f"{args.windows} spread val windows x {args.seq_len} bytes of `{args.data}` (bpic {payload['bpic']}), baseline val_bpb "
             f"**{base:.4f}**. Cell = % change in val_bpb at alpha {args.alpha}, beta 1-alpha, source rescaled to the destination norm "
             f"(2608.17981 eq. 1-2); negative is better. Rows: source block s; columns: destination block d.", "",
             "| s \\ d | " + " | ".join(str(d) for d in range(L - 1)) + " |", "|" + "---|" * L]
    grid = {(r["s"], r["d"]): r for r in results}
    for s in range(1, L):
        cells = []
        for d in range(L - 1):
            r = grid.get((s, d))
            cells.append(f"{r['pct']:+.2f}" if r else "")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", f"Best grid pair: s={best['s']} → d={best['d']}: {best['val_bpb']:.5f} ({best['pct']:+.2f} %)."]
    if scan:
        lines += ["", "Alpha scan on the best pair:", "", "| alpha | val_bpb | % |", "|---|---|---|"]
        lines += [f"| {r['alpha']} | {r['val_bpb']:.5f} | {r['pct']:+.2f} |" for r in sorted(scan + [best], key=lambda r: r["alpha"])]
    if variants:
        lines += ["", "Variants on the best pair:", "", "| variant | val_bpb | % |", "|---|---|---|"]
        lines += [f"| beta=1 | {variants[0]['val_bpb']:.5f} | {variants[0]['pct']:+.2f} |",
                  f"| ramp 10 chunks | {variants[1]['val_bpb']:.5f} | {variants[1]['pct']:+.2f} |"]
    lines += ["", f"Best overall: s={overall['s']} d={overall['d']} alpha={overall['alpha']} beta={overall['beta']} ramp={overall['ramp']}: "
              f"{overall['val_bpb']:.5f} ({overall['pct']:+.2f} %)."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}.json / .md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
