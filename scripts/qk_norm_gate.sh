#!/usr/bin/env bash
# QK-Norm in Relation: does it cost loss, and does τ_s / λ need to move with it (signed 2026-08-26).
#
# Why at all. 2608.24814 §4.2 finds QK-Norm and learnable RMSNorm gains are the two factors that decide how
# precisely ELR collapse holds — removing QK-Norm takes the collapse error from 2.3e-3 to 5.2e-3 — and ELR
# is now the coordinate Mote reads every norm-control gate on. Relation has learnable gains already;
# mote/model/relation.py did the evidence product on raw projections, with no QK-Norm anywhere.
#
# Why it needs a gate rather than a commit. In softmax attention QK-Norm is near-neutral, because softmax
# is scale-covariant. Relation's evidence is not: u = p1·p2ᵀ/√dh feeds silu(u) − λ·log i and sigmoid(u/τ_s).
# Measured on runs/t3l_dense_4e-4 after 12 h, feeding the trained w1/w2 an RMS-normalised stream:
#
#     u std per layer, unnormalised   1.37 1.17 0.91 0.81 0.69 0.61 0.59 0.60   (mean 0.84)
#     u std under QK-Norm             1.00 at every layer, times the learned gain²
#
# So it is a 1.18× change in the mean evidence scale — much smaller than feared — but it flattens a 2.3×
# spread across depth, which every layer's learned gain then has to re-establish if it wanted it. τ_s = 2.37
# reproduces the trained gate scale exactly; the arms bracket it either side.
#
#   bash scripts/qk_norm_gate.sh 2>&1 | tee docs/results/$(date +%F)-qk-norm.log
set -euo pipefail
OUT="${OUT:-runs/qk}"
mkdir -p "$OUT" docs/results
common=(--preset local --data data/local_mix --optimizer muon --lr 8e-4
        --batch-size 4 --seq-len 2048 --grad-accum 4
        --eval-every 500 --ckpt-minutes 30 --eval-spread --seed 42)

echo "== the pair: same everything, 2 h each, default τ_s 2.0 / λ 0.5 =="
python -m mote.cli train start -- "${common[@]}" --max-minutes 120 --out "$OUT/off"
python -m mote.cli train start -- "${common[@]}" --max-minutes 120 --qk-norm --out "$OUT/on"

echo "== the re-gate: τ_s either side of the scale-matching 2.37, and λ doubled, 1 h each =="
for T in 1.0 2.37 4.0; do
  python -m mote.cli train start -- "${common[@]}" --max-minutes 60 --qk-norm --tau-s "$T" \
      --out "$OUT/on_tau${T}"
done
python -m mote.cli train start -- "${common[@]}" --max-minutes 60 --qk-norm --lambda-init 1.0 \
    --out "$OUT/on_lam1.0"

python -m mote.cli train queue
echo
echo "Read the pair with scripts/elr_report.py (it prints Δ_coll against Mote's own seed noise), and the"
echo "re-gate on final val_bpb — 0.00025 bpb is seed noise there, so anything under ~0.0008 is not a result."
