"""How far the cached and windowed prompt reads drift from a cold one-shot read.

    python scripts/prefix_drift.py runs/<run>/last.pt --seq-len 16384 [--window 4096]

This is the measurement the windowed-prefill decision was signed against (2026-08-27). Two paths get
compared with the SAME cold one-shot read as the reference:

  windowed    prefill(first window) then forward_from_state over the rest — what serving now does
  warm        prefill(prefix) then forward_from_state over the suffix — what a cached continuation
              has always done, on every turn, since the prefix store shipped

The second column is the point. `Engine._verify_prefix` has existed since the store was built and
its `max_logit_diff` appears nowhere in docs/, so "windowing drifts by 8.6e-2" had nothing to be
compared against. A drift is only worth arguing about relative to the drift already being shipped.

What decides it is `boundary_flips`: the boundary sequence is what fills the arena, so a flip is a
different context, while a logit difference at the last position is float noise on one sample. The
signed bar is zero flips.

Writes docs/results/<date>-prefix-drift-<run>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mote.config import MoteConfig
from mote.model.hnet import HNetForCausalLM
from mote.runinfo import measured_bpic

ROOT = Path(__file__).resolve().parents[1]


def _read(model, arena, ids, device, window, split=0):
    """Read `ids`, optionally in `window`-sized pieces after a first `split` bytes. Returns
    (last logits, boundary mask, chunks)."""
    arena.invalidate()
    st = model.allocate_inference_state(device, arena=arena)
    L = len(ids)
    first = split or (window or L)
    spans = [(0, min(first, L))]
    rest = window or L
    spans += [(a, min(a + rest, L)) for a in range(spans[0][1], L, rest)]
    masks, lg = [], None
    with torch.no_grad():
        for k, (a, b) in enumerate(spans):
            w = torch.tensor([ids[a:b]], device=device)
            if k == 0:
                out = model.prefill(w, st, last_logits_only=True)
                lg, bm = out.logits[0, -1], out.routing.boundary_mask[0]
            else:
                seq, bm, _ = model.forward_from_state(w, st, last_logits_only=True)
                lg = seq[0, -1]
            masks.append(bm)
    return lg.float(), torch.cat(masks), st.main.n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--window", type=int, default=None, help="default: the checkpoint's cfg.prefill_window")
    ap.add_argument("--data", default="data/flagship_mix")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ck = torch.load(Path(args.ckpt), map_location="cpu", weights_only=True)
    cfg = MoteConfig.from_dict(ck["config"])
    cfg.mbp.enabled = False
    model = HNetForCausalLM(cfg)
    model.load_state_dict(ck["model"], strict=False)
    model.to(args.device).eval()
    window = args.window or cfg.prefill_window
    L = min(args.seq_len, cfg.max_seq_len)

    data = args.data if Path(f"{args.data}.meta.json").exists() else "data/local_mix"
    from mote.data.loader import ByteShard

    shard = ByteShard(data, "val")
    gen = torch.Generator().manual_seed(0)
    bpic = measured_bpic(args.ckpt)
    arena = model.new_arena(args.device, bpic=bpic)

    rows = []
    V = cfg.vocab_size
    for i in range(args.samples):
        ids = shard.sample_batch(1, L, gen)[0][0].tolist()
        t0 = time.perf_counter()
        ref_lg, ref_bm, ref_n = _read(model, arena, ids, args.device, window=0)
        ref_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        win_lg, win_bm, win_n = _read(model, arena, ids, args.device, window=window)
        win_ms = (time.perf_counter() - t0) * 1000
        # the warm path: a cached prefix, then the rest — one split, wherever a turn boundary would be
        warm_lg, warm_bm, warm_n = _read(model, arena, ids, args.device, window=0, split=L // 2)

        def cmp(lg, bm, n, ms):
            return {
                "boundary_flips": int((bm != ref_bm).sum()),
                "chunks": n, "chunks_cold": ref_n,
                "max_logit_diff": float((lg[:V] - ref_lg[:V]).abs().max()),
                "ms": round(ms, 1),
            }

        rows.append({"sample": i, "bytes": L,
                     "windowed": cmp(win_lg, win_bm, win_n, win_ms),
                     "warm_continuation": cmp(warm_lg, warm_bm, warm_n, 0.0),
                     "cold_ms": round(ref_ms, 1)})
        r = rows[-1]
        print(f"sample {i}: cold {ref_n} chunks in {ref_ms:.0f} ms | "
              f"windowed({window}) flips={r['windowed']['boundary_flips']} "
              f"dlogit={r['windowed']['max_logit_diff']:.2e} {win_ms:.0f} ms | "
              f"warm flips={r['warm_continuation']['boundary_flips']} "
              f"dlogit={r['warm_continuation']['max_logit_diff']:.2e}", flush=True)

    res = {"ckpt": str(args.ckpt), "seq_len": L, "window": window, "measured_bpic": bpic,
           "device": args.device, "data": data, "samples": rows,
           "worst": {k: max(r[k]["boundary_flips"] for r in rows) for k in ("windowed", "warm_continuation")}}
    out = Path(args.out) if args.out else (
        ROOT / "docs/results" / f"{time.strftime('%Y-%m-%d')}-prefix-drift-{Path(args.ckpt).parent.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"\nworst boundary flips — windowed {res['worst']['windowed']}, "
          f"warm continuation {res['worst']['warm_continuation']}  (the bar is 0 for windowed)")
    print("written:", out)


if __name__ == "__main__":
    main()
