"""Router health of an MoE checkpoint beyond balance: expert usage per layer on each per-domain val slice,
normalised mutual information I(E;D)/H(D) and mean pairwise Jensen-Shannon divergence between the domains'
routing distributions (Kakao 2608.20061 App. G: healthy routers are balanced AND specialised — MI/JSD rise
with depth; a balanced-but-domain-blind router is the failure this catches, 2608.21236 §3), plus MaxVio.

    python -m mote.eval.moe_report --checkpoint runs/moe_e8/last.pt [--domains data/flagship_val] [--batches 32]

Writes moe_report.json next to the checkpoint. Router entropy is deliberately not reported as a confidence
signal (2608.17687: its sign flips between models).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import torch

from ..model.moe import moe_modules
from .val_bpb import load_model, raw_shard


def _entropy(p: torch.Tensor) -> float:
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def mi_and_jsd(usage: torch.Tensor) -> Dict[str, float]:
    """`usage` [n_domains, E], rows sum to 1 (routing distribution per domain, uniform domain prior)."""
    n = usage.shape[0]
    if n < 2:
        return {"nmi": 0.0, "jsd": 0.0}
    marginal = usage.mean(0)
    h_e = _entropy(marginal)
    h_e_given_d = sum(_entropy(usage[i]) for i in range(n)) / n
    nmi = (h_e - h_e_given_d) / max(math.log(n), 1e-9)  # I(E;D) / H(D)
    js, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            m = 0.5 * (usage[i] + usage[j])
            js += 0.5 * _kl(usage[i], m) + 0.5 * _kl(usage[j], m)
            pairs += 1
    return {"nmi": float(nmi), "jsd": js / max(pairs, 1)}


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    mask = p > 0
    return float((p[mask] * (p[mask] / q[mask].clamp_min(1e-12)).log()).sum())


@torch.no_grad()
def usage_on(model, shard, batches: int, seq_len: int, device) -> List[torch.Tensor]:
    """Per-layer expert usage (mean load over windows) on a shard: [layers][E]."""
    mods = moe_modules(model)
    acc = [torch.zeros(m.n_experts) for m in mods]
    n = 0
    for batch, _ in shard.sequential_batches(1, seq_len, batches, spread=True):
        batch = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model(batch[:, :-1])
        for i, m in enumerate(mods):
            acc[i] += m.stats["load"].float().cpu()
        n += 1
    return [a / max(n, 1) for a in acc]


def run(checkpoint: str | Path, domains: str | Path, batches: int, seq_len: int | None, device) -> Dict:
    model, cfg, step = load_model(checkpoint, device)
    seq_len = seq_len or cfg.max_seq_len
    mods = moe_modules(model)
    out: Dict = {"checkpoint": str(checkpoint), "step": step, "layers": len(mods), "experts": mods[0].n_experts if mods else 0, "domains": {}, "per_layer": []}
    if not mods:
        return out
    per_domain: Dict[str, List[torch.Tensor]] = {}
    for p in sorted(Path(domains).glob("*.val.bin")):
        name = p.name.split(".")[0]
        per_domain[name] = usage_on(model, raw_shard(p), batches, seq_len, device)
        out["domains"][name] = [[round(float(v), 4) for v in layer] for layer in per_domain[name]]
    names = list(per_domain)
    for li in range(len(mods)):
        u = torch.stack([per_domain[d][li] for d in names])  # [domains, E]
        u = u / u.sum(1, keepdim=True).clamp_min(1e-12)
        overall = u.mean(0)
        maxvio = float((overall.max() - overall.mean()) / overall.mean().clamp_min(1e-12))
        out["per_layer"].append({"layer": li, "maxvio": maxvio, **mi_and_jsd(u)})
    out["summary"] = {
        "maxvio_mean": sum(r["maxvio"] for r in out["per_layer"]) / len(mods),
        "nmi_mean": sum(r["nmi"] for r in out["per_layer"]) / len(mods),
        "nmi_last": out["per_layer"][-1]["nmi"],
        "jsd_mean": sum(r["jsd"] for r in out["per_layer"]) / len(mods),
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--domains", default="data/flagship_val")
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    res = run(args.checkpoint, args.domains, args.batches, args.seq_len, device)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "moe_report.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "domains"}, indent=1))


if __name__ == "__main__":
    main()
