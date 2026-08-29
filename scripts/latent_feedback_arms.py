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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The run.json keys that ARE the trunk's recipe (everything else in it is the run's own: out, resume, max
# minutes, snapshots). The flag for each comes from the trainer's parser, so a renamed flag fails here
# instead of silently dropping out of every derived arm.
RECIPE = ["preset", "config", "data", "seq_len", "batch_size", "grad_accum", "bucket", "optimizer", "weight_decay",
          "lr", "clip", "beta2", "eval_batches", "eval_every", "bound_floor", "bound_ceiling", "seed", "ckpt_minutes",
          "ckpt_main", "eval_spread", "tf32", "qk_norm"]


def recipe_flags(run: dict) -> list:
    from mote.train.train import build_argparser

    actions = {a.dest: a for a in build_argparser()._actions}
    missing = [d for d in RECIPE if d not in actions]
    if missing:
        raise SystemExit(f"run.json keys with no trainer flag any more: {missing}")
    out = []
    for d in RECIPE:
        v = run.get(d)
        if v is None or v is False:
            continue
        flag = actions[d].option_strings[0]
        out += [flag] if actions[d].nargs == 0 else [flag, str(v)]
    return out


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
    common = recipe_flags(run)
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
