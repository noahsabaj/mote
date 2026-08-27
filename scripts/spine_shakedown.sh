#!/usr/bin/env bash
# ~1 h at the real shape before committing 24 (signed 2026-08-27).
#
# Nothing has ever completed a training step at Mote-96M or Mote-138M. runs/flagship_shape_v2 was
# configured on 2026-08-24 at 16384 with ckpt_main false and has a 0-byte log — it never produced a
# step. Every number we have at these shapes (4.31 GB, 68.1 KB/s) is from the profiler, not a run.
#
# So A0 would be the first arm ever at this shape AND one leg of a three-way comparison. If the
# shape itself is wrong, all three arms fail together and the gate says nothing about the spine.
# This settles four things the gate depends on and cannot check for itself:
#
#   does it train           loss falls, no norm-guard trip, no OOM at an eval or a checkpoint
#   real throughput         the profiler's B/s is not a training run's; the ELR match and the
#                           matched-token read both key off step counts
#   does expand fit         at 16384 with --ckpt-main, measured rather than extrapolated — the
#                           138M extrapolation already underestimated once
#   do the diagnostics run  stream_cos, h_res_drift and alpha_res have to appear in the log, or
#                           the gate produces numbers nobody can interpret
#
#   bash scripts/spine_shakedown.sh 2>&1 | tee docs/results/$(date +%F)-spine-shakedown.log
set -euo pipefail
OUT="${OUT:-runs/shakedown}"
MIN="${MIN:-20}"
DATA="${DATA:-data/flagship_mix}"
[ -f "$DATA.meta.json" ] || DATA="data/local_mix"
mkdir -p "$OUT" docs/results

for SPINE in off frac expand; do
  echo "== $SPINE =="
  python -m mote.train.profile_step --preset mote-96m --data "$DATA" --ckpt-main \
      --batch-size 1 --seq-len 16384 --chunk-bytes 6 --spine "$SPINE" \
      --out "$OUT/profile_$SPINE.json" \
    || { echo "   $SPINE does not fit at 96M/16384 — the gate cannot run this arm"; continue; }
  python -m mote.cli train start -- \
      --preset mote-96m --data "$DATA" --optimizer muon --lr 8e-4 --ckpt-main \
      --batch-size 1 --grad-accum 4 --seq-len 16384 --bucket 64 \
      --schedule trunk --eval-ema 0.9999 \
      --max-minutes "$MIN" --eval-every 200 --eval-batches 4 --eval-spread \
      --log-every 50 --ckpt-minutes 999 --no-mbp --seed 42 \
      --spine "$SPINE" $([ "$SPINE" = off ] || echo --spine-n 4) --out "$OUT/$SPINE"
done

python -m mote.cli train queue
cat <<'NEXT'

When they finish, check all four before queueing the gate:
  python - <<'EOF'
  import json, pathlib
  for d in sorted(pathlib.Path("runs/shakedown").glob("*/log.jsonl")):
      rs = [json.loads(l) for l in d.read_text().splitlines()]
      ev = [r["eval"]["val_bpb"] for r in rs if "eval" in r]
      lg = [r for r in rs if "train_bpb" in r]
      dg = next((r for r in reversed(lg) if "stream_cos" in r), None)
      print(f"{d.parent.name:8s} evals {len(ev)} first {ev[0]:.3f} last {ev[-1]:.3f} "
            f"B/s {lg[-1]['bytes_per_sec']:.0f} peak {rs[-1]['peak_gb']:.2f} GB "
            f"cos {dg['stream_cos'] if dg else '—'}")
  EOF
NEXT
