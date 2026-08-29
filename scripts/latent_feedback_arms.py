#!/usr/bin/env python
"""Queue the three latent-feedback arms (docs/results/2026-08-28-latent-feedback-prereg.md, signed 2026-08-28).

    PYTHONPATH=. .venv/bin/python scripts/latent_feedback_arms.py runs/trunk/snap_090000.pt [--minutes 1440] [--dry-run]

Reads the trunk's run.json beside the snapshot for the recipe flags it must mirror (preset, data, window,
batch, accumulation, bucket, optimizer, weight decay, the constant lr, bounded routing, eval settings) and
submits `control`, `chunk` and `byte` as continuations from the snapshot — the pre mix, the trunk's
constant lr, 24 h each — with the feedback arms on the paper's 75/22/3 pass mixture and every arm
evaluated with two fused prefill passes (the gate reads val_bpb_fb1 against the control's val_bpb).
The byte arm's memory fallbacks (--feedback-window 8192, then --feedback-detach) are applied by hand
after the profile, never by default here."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MIRROR = {  # run.json key -> flag (value flags); the trunk's recipe, nothing else
    "preset": "--preset", "config": "--config", "data": "--data", "seq_len": "--seq-len", "batch_size": "--batch-size",
    "grad_accum": "--grad-accum", "bucket": "--bucket", "optimizer": "--optimizer", "weight_decay": "--weight-decay",
    "lr": "--lr", "clip": "--clip", "beta2": "--beta2", "eval_batches": "--eval-batches", "eval_every": "--eval-every",
    "bound_floor": "--bound-floor", "bound_ceiling": "--bound-ceiling", "seed": "--seed", "ckpt_minutes": "--ckpt-minutes",
}
MIRROR_FLAGS = {"ckpt_main": "--ckpt-main", "eval_spread": "--eval-spread", "tf32": "--tf32", "qk_norm": "--qk-norm"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", help="the trunk snapshot checkpoint (.pt); its run.json sits beside it")
    ap.add_argument("--out", default="runs/lf", help="output prefix: <out>/control, <out>/chunk, <out>/byte")
    ap.add_argument("--minutes", type=float, default=1440.0)
    ap.add_argument("--mix", default="0.75,0.22,0.03")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    snap = Path(args.snapshot)
    run = json.loads((snap.parent / "run.json").read_text())
    common = []
    for k, flag in MIRROR.items():
        v = run.get(k)
        if v is None or v is False:
            continue
        common += [flag, str(v)]
    common += [flag for k, flag in MIRROR_FLAGS.items() if run.get(k)]
    common += ["--init-from", str(snap), "--schedule", "constant", "--max-minutes", str(args.minutes),
               "--eval-feedback-passes", "2"]
    arms = {"control": [], "chunk": ["--feedback", "chunk", "--feedback-mix", args.mix], "byte": ["--feedback", "byte", "--feedback-mix", args.mix]}
    for name, extra in arms.items():
        argv = [sys.executable, "-m", "mote.cli", "train", "start", "--", *common, *extra, "--out", f"{args.out}/{name}"]
        print(" ".join(argv[3:]), flush=True)
        if not args.dry_run:
            subprocess.run(argv, check=True)
    if not args.dry_run:
        subprocess.run([sys.executable, "-m", "mote.cli", "train", "queue"], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
