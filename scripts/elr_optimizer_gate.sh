#!/usr/bin/env bash
# Muon vs Muon-SW, decided in ELR instead of in nominal lr (signed 2026-08-26).
#
# The freeze picked Muon on 1.17734 vs 1.18006 — a 0.00272 bpb gap, and seed noise on that metric is
# 0.00025 bpb, so the measurement is precise. It is also unattributable. Muon-SW is not a different
# optimizer: mote/train/muon.py differs in one line, the decoupled decay, which makes it a NORM-CONTROL
# variant. Measured on the two gate checkpoints, it ran at 0.914x Muon's ELR at the same lr 8e-4, and the
# local slope d(val_bpb)/d(ln ELR) flips sign with horizon (-0.073 bpb/nat at 30 min, +0.026 at 12 h), so
# that ELR gap explains between 0.000 and 0.0066 bpb of a 0.00272 bpb effect.
#
# 2608.24814 App. B.2's reference-run protocol settles it: leave Muon-SW's decay in place and adapt only
# its per-matrix learning rates to track Muon's recorded ELR. Then the remaining gap is the update rule
# and nothing else.
#
# Two outcomes, both worth the 4 h:
#   collapse to ~1e-3  the law holds on Mote, and Muon-SW offers nothing an ELR schedule cannot
#   no collapse        the law does NOT hold at Mote's architecture, which we need to know before
#                      reading every future norm-control gate on the ELR coordinate
#
#   bash scripts/elr_optimizer_gate.sh 2>&1 | tee docs/results/$(date +%F)-elr-optimizer-gate.log
set -euo pipefail
OUT="${OUT:-runs/elr_gate}"
MIN="${MIN:-120}"
LR="${LR:-8e-4}"
mkdir -p "$OUT" docs/results

common=(--preset local --data data/local_mix --optimizer muon --lr "$LR"
        --batch-size 4 --seq-len 2048 --grad-accum 4 --max-minutes "$MIN"
        --eval-every 500 --ckpt-minutes 30 --eval-spread --seed 42)

echo "== 1/2  the reference: Muon, recording its per-matrix ELR =="
python -m mote.cli train start -- "${common[@]}" \
    --elr-trace-out elr_trace.json --out "$OUT/muon_ref"

echo "== 2/2  Muon-SW keeping its own decay, tracking that ELR =="
# --optimizer muonsw is overridden below by --elr-match only for the LEARNING RATES; the eta^2-scaled
# decay stays in force, which is exactly the point: the norm control differs, the ELR does not.
python -m mote.cli train start -- "${common[@]}" \
    --optimizer muonsw --elr-match "$OUT/muon_ref/elr_trace.json" --out "$OUT/muonsw_matched"

python -m mote.cli train queue
echo
echo "When they finish:  python scripts/elr_report.py $OUT/muon_ref $OUT/muonsw_matched"
