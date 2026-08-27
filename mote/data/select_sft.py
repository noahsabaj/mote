"""Difficulty-select an SFT shard: keep the windows that are hard for the base *and* learnable.

    python -m mote.data.select_sft --shard data/sft_local --split train --base runs/t3l_dense_8e-4/last.pt \
        --calibrated runs/sft_warmup/last.pt --seq-len 2048 --keep-frac 0.4 --out data/sft_local.keep.npy

Two 2026 results say the SFT mix is worth choosing rather than taking whole:

* 2603.01293 — "SFT learns best from a small set of examples challenging for the pretrained model, while
  excessively large SFT datasets may dilute informative pretraining signals." So a low base loss means the
  window teaches nothing the base does not already do, and including it costs pretraining signal.
* 2601.23006 (InstructDiff) — rank by the *difference* between the base and a briefly instruction-tuned
  calibration of it, not by either alone; 10% of the data beat 100% of it. Loss that stays high after a
  short warm-up is noise the model cannot learn; loss that drops is a lesson it can.

So a window is worth keeping when the base finds it hard AND a little tuning already helps:

    L_base   mean NLL per byte over the window's assistant bytes under --base
    L_cal    the same under --calibrated (a short warm-up SFT of the base; omit to rank on L_base alone)
    delta    L_base - L_cal, how much a little tuning bought
    score    --by delta | loss | product (default product: hard *and* learnable, each rank-normalised)

Writes an int64 .npy of the kept window starts, which ByteShard(..., keep=...) samples from instead of
sampling the shard uniformly, plus a .json report of the distribution. NOT free: a 127 MB shard at
seq-len 2048 is ~62k windows, about 4 minutes of forward passes on the 4060 Ti and hours on CPU — run it
on the card once the queue drains, or pass --max-windows to score a sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..config import MoteConfig
from ..model.hnet import HNetForCausalLM
from .loader import ByteShard


def _load(path: str, device) -> HNetForCausalLM:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = HNetForCausalLM(MoteConfig.from_dict(ck["config"]), device=device)
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def window_losses(model: HNetForCausalLM, shard: ByteShard, starts: List[int], seq_len: int,
                  batch_size: int, device) -> np.ndarray:
    """Mean NLL per *masked* byte for each window; nan where a window has no assistant bytes."""
    out = np.full(len(starts), np.nan, dtype=np.float64)
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        wins = [shard._window(s, seq_len) for s in chunk]
        ids = torch.from_numpy(np.stack([w[0] for w in wins])).to(device)
        m = torch.from_numpy(np.stack([w[1] for w in wins])).to(device).float()
        inputs, targets, tmask = ids[:, :-1], ids[:, 1:], m[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(inputs).logits.float()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").view(targets.shape)
        n = tmask.sum(-1)
        tot = (nll * tmask).sum(-1)
        vals = torch.where(n > 0, tot / n.clamp_min(1), torch.full_like(tot, float("nan")))
        out[i : i + len(chunk)] = vals.double().cpu().numpy()
    return out


def _ranks(x: np.ndarray) -> np.ndarray:
    """Rank-normalise to [0,1], nan-safe (nans rank last)."""
    order = np.argsort(np.where(np.isnan(x), np.inf, x))
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x)) / max(len(x) - 1, 1)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="prefix, e.g. data/sft_local")
    ap.add_argument("--split", default="train")
    ap.add_argument("--base", required=True, help="the pretrained checkpoint SFT-1 would start from")
    ap.add_argument("--calibrated", default=None, help="a short warm-up SFT of --base; omit to rank on the base loss alone")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--keep-frac", type=float, default=0.4, help="share of scorable windows to keep")
    ap.add_argument("--by", default="product", choices=["product", "delta", "loss"])
    ap.add_argument("--max-windows", type=int, default=0, help="score an evenly spread sample of this many windows (0 = all)")
    ap.add_argument("--out", default=None, help="default: {shard}.keep.npy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    shard = ByteShard(args.shard, args.split, sft=True)
    n_windows = (shard.n - 1) // args.seq_len
    starts = [j * args.seq_len for j in range(n_windows)]
    if args.max_windows and args.max_windows < n_windows:
        picks = [round(k * (n_windows - 1) / max(args.max_windows - 1, 1)) for k in range(args.max_windows)]
        starts = [j * args.seq_len for j in picks]
    print(json.dumps({"windows": len(starts), "of": n_windows, "device": str(device)}), flush=True)

    base = _load(args.base, device)
    l_base = window_losses(base, shard, starts, args.seq_len, args.batch_size, device)
    del base
    if args.calibrated:
        cal = _load(args.calibrated, device)
        l_cal = window_losses(cal, shard, starts, args.seq_len, args.batch_size, device)
        del cal
        delta = l_base - l_cal
    else:
        l_cal = np.full_like(l_base, np.nan)
        delta = np.zeros_like(l_base)

    scorable = ~np.isnan(l_base)
    if args.by == "loss" or not args.calibrated:
        score = _ranks(l_base)                       # hard for the base
    elif args.by == "delta":
        score = _ranks(delta)                        # most improved by a little tuning
    else:
        score = _ranks(l_base) * _ranks(delta)       # hard AND learnable
    score = np.where(scorable, score, -np.inf)

    k = max(1, int(round(args.keep_frac * int(scorable.sum()))))
    keep_idx = np.argsort(-score)[:k]
    keep = np.array(sorted(int(starts[i]) for i in keep_idx), dtype=np.int64)

    out = Path(args.out) if args.out else Path(f"{args.shard}.keep.npy")
    np.save(out, keep)
    rep = {"shard": args.shard, "split": args.split, "seq_len": args.seq_len, "by": args.by,
           "scored": len(starts), "scorable": int(scorable.sum()), "kept": len(keep), "keep_frac": args.keep_frac,
           "base": args.base, "calibrated": args.calibrated,
           "l_base": {"mean": float(np.nanmean(l_base)), "p10": float(np.nanpercentile(l_base, 10)),
                      "p50": float(np.nanpercentile(l_base, 50)), "p90": float(np.nanpercentile(l_base, 90))},
           "kept_l_base_mean": float(np.nanmean(l_base[keep_idx])),
           "dropped_l_base_mean": float(np.nanmean(np.delete(l_base, keep_idx))),
           "out": str(out)}
    if args.calibrated:
        rep["delta"] = {"mean": float(np.nanmean(delta)), "p50": float(np.nanpercentile(delta, 50))}
        rep["kept_delta_mean"] = float(np.nanmean(delta[keep_idx]))
    Path(str(out) + ".json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
