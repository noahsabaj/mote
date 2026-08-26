#!/usr/bin/env bash
# The 3x3: three arms trained at 5/15/30 % failure, each probed at all three rates (signed 2026-08-26).
#
# Why a matrix and not three numbers. The probe should carry the same failure rate as the training data —
# a probe whose worlds cannot fail cannot measure whether the model tracks a world that can. But then each
# arm is graded on its own exam and they stop being comparable. The matrix keeps both:
#
#   the DIAGONAL     does training at rate r help at rate r
#   the OFF-DIAGONAL does training at 30 % transfer down to 5 %, and vice versa
#
# The off-diagonal is the one that decides. If a high rate transfers down, take the high rate: it teaches
# recovery and costs nothing on easier worlds. If it does not, the rate is a genuine trade and the middle
# is the safe pick.
#
# CPU only — the arms already ran; this is nine probe passes plus three recovery passes.
#
#   bash scripts/pfail_matrix.sh 2>&1 | tee docs/results/$(date +%F)-p-fail.md
set -euo pipefail
OUT="${OUT:-runs/pfail}"
N="${N:-120}"
for PF in 5 15 30; do
  [ -f "$OUT/pf_$PF/last.pt" ] || { echo "missing $OUT/pf_$PF/last.pt — run scripts/queue_thursday.sh first"; exit 1; }
done

for TRAIN in 5 15 30; do
  for PROBE in 5 15 30; do
    python -m mote.eval.sim_probe --checkpoint "$OUT/pf_$TRAIN/last.pt" --n "$N" --p-fail "$PROBE" \
        --out "$OUT/pf_${TRAIN}/probe_at_${PROBE}.json" >/dev/null
  done
  # recovery is what the failures are FOR; it needs no probe rate of its own (its worlds always fail)
  python -m mote.eval.recovery_probe --checkpoint "$OUT/pf_$TRAIN/last.pt" --n 60 \
      --out "$OUT/pf_${TRAIN}/recovery.json" >/dev/null
done

python scripts/pfail_report.py
