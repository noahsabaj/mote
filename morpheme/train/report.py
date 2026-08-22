"""Summarize a run's log.jsonl in the terminal: curves as sparklines + the latest chunk samples.

    python -m morpheme.train.report runs/pilot_1h
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BARS = "▁▂▃▄▅▆▇█"


def spark(xs, width=60):
    xs = [x for x in xs if x is not None]
    if not xs:
        return ""
    if len(xs) > width:  # downsample by averaging buckets
        k = len(xs) / width
        xs = [sum(xs[int(i * k): int((i + 1) * k)]) / max(len(xs[int(i * k): int((i + 1) * k)]), 1) for i in range(width)]
    lo, hi = min(xs), max(xs)
    rng = (hi - lo) or 1.0
    return "".join(BARS[min(int((x - lo) / rng * (len(BARS) - 1)), len(BARS) - 1)] for x in xs)


def main(run: str):
    try:  # Windows consoles default to cp1252, which cannot print the sparkline glyphs
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    log = Path(run) / "log.jsonl"
    recs = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    train = [r for r in recs if "train_bpb" in r]
    evals = [r for r in recs if "eval" in r]
    print(f"{run}: {len(train)} train records, {len(evals)} evals, last step {recs[-1].get('step')} at {recs[-1].get('elapsed_min', 0):.1f} min")
    for key, label in [("train_bpb", "train bpb"), ("bpic", "bytes/chunk"), ("ce_mbp", "mbp ce"), ("bytes_per_sec", "bytes/s"), ("grad_norm", "grad norm"), ("lr", "lr"), ("target_ratio", "target N")]:
        xs = [r.get(key) for r in train]
        if any(x is not None for x in xs):
            vals = [x for x in xs if x is not None]
            print(f"  {label:12s} {spark(xs)}  {vals[0]:.4g} → {vals[-1]:.4g}")
    if evals:
        for key, label in [("val_bpb", "val bpb"), ("val_bpic", "val bytes/chunk"), ("boundary_on_separator_frac", "boundary@sep"), ("mbp_top1_acc", "mbp top-1")]:
            xs = [r["eval"].get(key) for r in evals]
            vals = [x for x in xs if x is not None]
            if vals:
                print(f"  {label:16s} {spark(xs, 40)}  {vals[0]:.4g} → {vals[-1]:.4g}")
        print("  chunk samples:")
        for r in evals[-3:]:
            print(f"    step {r['step']:>6}: {r['eval'].get('sample', '')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/pilot_1h")
