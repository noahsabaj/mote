#!/usr/bin/env bash
# Everything waiting on the card, queued behind the current run (signed 2026-08-26).
#
# Two independent questions, four jobs, ~3 GPU-hours total. Both are local — nothing here spends a
# Lightning credit, and nothing needs supervision: the daemon runs them in order and you read the two
# results together.
#
#   Round A      which preference objective, and does the fixed identity data stop the false firing
#                (docs/research/dpo-rlvr-2026-08-25.md; the whole merged post-training stage waits on it)
#   p_fail arms  how often an action should fail in the sim, which gates the regeneration and therefore
#                the flagship's mid-training data
#
# The p_fail question cannot be answered by a probe on a stale checkpoint — the first version of
# scripts/sweep_p_fail.sh tried, and it would have printed the same number for every rate, because
# sim_probe built its held-out worlds at p_fail=0 whatever the training data did. It needs arms.
#
#   bash scripts/queue_thursday.sh 2>&1 | tee docs/results/$(date +%F)-queue-thursday.log
set -euo pipefail
BASE="${BASE:-runs/t3l_dense_4e-4/last.pt}"   # the best of the three 12-h LR arms (val_bpb_ema 1.0276)
OUT="${OUT:-runs/pfail}"
mkdir -p "$OUT" docs/results
[ -f "$BASE" ] || { echo "no $BASE — check the LR sweep finished"; exit 1; }

echo "== the three p_fail arms (30 min each) =="
# Same data, same length, same everything: only the failure rate in the sim slice differs. The sim share
# is deliberately large (25 %) so 30 minutes can see the difference at all — this is a screening arm for
# the RATE, not a candidate checkpoint.
for PF in 5 15 30; do
  [ -f "data/pfail/plain_${PF}.meta.json" ] || { echo "missing data/pfail/plain_${PF} — build it first"; exit 1; }
  python -m mote.cli train start -- \
      --preset local --init-from "$BASE" --data data/local_mix \
      --mix "data/pfail/plain_${PF}:0.25" \
      --optimizer muon --lr 3e-4 --batch-size 4 --grad-accum 4 --seq-len 2048 \
      --max-minutes 30 --eval-every 500 --ckpt-minutes 10 --no-mbp --eval-spread \
      --out "$OUT/pf_$PF"
done

echo "== Round A: its two shared SFTs, then the seven preference arms run on CPU-cheap time =="
python -m mote.data.build_identity --out data/ra_id_plain   --params 31643528 --n 4000 --pairs 400 --swap 200 --neg 200 --ties 200 --neutral-frac 0
python -m mote.data.build_identity --out data/ra_id_neutral --params 31643528 --n 4000 --pairs 400 --swap 200 --neg 200 --ties 200 --neutral-frac 0.15
cp data/ra_id_plain.dpo.jsonl runs/roundA_pairs.jsonl 2>/dev/null || { mkdir -p runs; cp data/ra_id_plain.dpo.jsonl runs/roundA_pairs.jsonl; }
for MIX in plain neutral; do
  python -m mote.cli train start -- \
      --preset local --init-from "$BASE" --sft --data "data/ra_id_${MIX}" \
      --optimizer muon --lr 3e-4 --batch-size 4 --grad-accum 4 --seq-len 2048 \
      --max-minutes 30 --eval-every 500 --ckpt-minutes 10 --no-mbp --eval-spread \
      --out "runs/roundA/sft_$MIX"
done

python -m mote.cli train queue
echo
echo "Queued. When they finish:"
echo "  bash scripts/pfail_matrix.sh          # the 3x3, on CPU"
echo "  bash scripts/round_a.sh --skip-sft    # the seven preference arms (~3 min each) + the table"
