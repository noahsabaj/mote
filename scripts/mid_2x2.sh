#!/usr/bin/env bash
# Mid-training: mixture x decay, plus a floor. Signed 2026-08-26 (docs/shape.md § mid,
# docs/research/midtraining-2026-08-26.md).
#
# What this answers that the old single-axis gate could not:
#
#   1. Does the ANNEAL reweighting beat the flagship composition?  -> mix C vs mix B
#   2. Does the decay earn its 20 % of the branch at all?          -> decayed vs constant
#
# The old design answered neither cleanly. Its two branches were both cooldowns, so nothing tested the
# decay; and only the anneal carried the sim/chat/identity extras, so every one of its three deciders was
# won by data inclusion rather than by the mixture. The extras are in BOTH branches here, which is what
# makes the A/B about the web half alone.
#
#   fork at 80 %:            +--- --schedule branch,   decay 80->100 %  -> *_decayed
#   constant 8e-4 to 80 % ---+
#                            +--- --schedule constant, 80->100 %        -> *_constant   (token-matched)
#
# Token matching is why the constant arm is a second run and not just the 80 % snapshot: the snapshot has
# seen 20 % fewer bytes than the decayed endpoint, and the decay axis would be confounded by data volume.
#
# Budget: 2 x 1.4 + 2 x 0.28 = 3.36 GPU-days of branches, 5 x 60-min SFT, ~3.6 GPU-days total. Against
# 2.88 for the old 1x2. The card must be free; this queues the branches through the daemon and waits.
#
#   bash scripts/mid_2x2.sh <TRUNK_SNAPSHOT> 2>&1 | tee docs/results/$(date +%F)-mid-2x2.log
set -euo pipefail
SNAP="${1:?usage: mid_2x2.sh runs/trunk/snap_XXXXXXXX.pt}"
PARAMS="${PARAMS:-100000000}"
MIN="${MIN:-2016}"       # minutes per full branch (~1.4 GPU-days)
TAIL=$(( MIN / 5 ))      # the last 20 %
OUT=runs/mid
mkdir -p "$OUT" docs/results

P_FAIL="${P_FAIL:?set P_FAIL after sweeping 5/15/30 against the sim probe — see step 0b}"

echo "== 0. regenerate the sim. Signed 2026-08-26: until then no action could fail, which made 99.7 % of"
echo "      every tool result a restatement of its call, and left the environment's three refusal strings"
echo "      absent from all 20,000 expert traces (docs/research/midtraining-2026-08-26.md)."
python -m mote.sim.generate --out data/sim_state --mb 150 --dpo-pairs 20000 \
    --p-fail "$P_FAIL" --parallel-frac 0.2 --swap-frac 0.3
python -m mote.data.build_local --out data/sim_plain --text data/sim_state.narrative.jsonl --chat data/sim_state.qa.jsonl
python -m mote.data.build_local --out data/sim_sft   --chat data/sim_state.qa.jsonl --sft
python -m mote.sim.long --out data/sim_long --mb 320 --min-bytes 4000 --max-bytes 16000
python -m mote.sim.counterfactual --out data/sim_cf --n 20000 --p-fail "$P_FAIL"
# Expert traces WITH recovery: until 2026-08-26 not one of 20,000 contained a refusal of any kind, so
# RLVR-1 would have met its first one having never seen it (2608.20314, 2607.12463).
python -m mote.sim.tasks --out data/sim_traces --n 20000 --recover-frac 0.35
python -m mote.data.build_local --out data/sim_traces --chat data/sim_traces.jsonl --sft
python -m mote.data.build_local --out data/sim_long_plain --text data/sim_long.jsonl data/sim_cf.jsonl

echo "== 0b. the extras — identical in both branches, so they are not a variable"
# --typo-frac is 2606.16246's answer to repetition applied where the repetition is: 3 % of an 8 GB branch
# is ~240 MB generated from about 12 KB of distinct prose.
python -m mote.data.build_spec_docs --out data/spec_docs.jsonl --n 400000 --params "$PARAMS" \
    --typo-frac 0.3 --typo-rate 0.012 --spec-out data/mote-spec.md
python -m mote.data.build_local --out data/spec_plain --text data/spec_docs.jsonl

# The extras budget, 15 % of each branch (docs/shape.md § mid). Identity is documents now, not Q&A: the
# card recited as an answer is what produced identity_recite_rate 0.70 before DPO ever ran.
EXTRAS=(--mix data/spec_plain:0.03:plain
        --mix data/sim_traces:0.03:fim
        --mix data/sim_long_plain:0.04:plain
        --mix data/sft_local:0.05:plain)

# --min-lr-ratio is left at its per-schedule default (0 for branch/constant); --init-from differs per arm
COMMON=(--preset flagship --optimizer muon --batch-size 1 --grad-accum 4 --seq-len 16384
        --eval-spread --eval-every 2000 --eval-batches 16 --log-every 100 --ckpt-minutes 15 --lr 8e-4)

# No --serve on any arm. The standing rule pins the trunk and the branch that becomes the flagship base,
# but four arms cannot all be on the air and picking one before the gate has run would prejudge it. The
# daemon keeps serving whatever is pinned; pin the winner by hand after step 5.
echo "== 1. the two branches, each forking at 80 %"
for MIX in b:data/flagship_mix_b c:data/anneal_mix_c; do
  NAME="${MIX%%:*}"; DATA="${MIX##*:}"
  python -m mote.cli train start -- "${COMMON[@]}" "${EXTRAS[@]}" --init-from "$SNAP" \
      --data "$DATA" --schedule branch --max-minutes "$MIN" --snapshot-at 0.8 \
      --out "$OUT/${NAME}_decayed"
done
python -m mote.cli train queue   # both branches run to completion before the forks are submitted

echo "== 2. the no-decay arms, resumed from each 80 % snapshot for the remaining bytes"
for NAME in b c; do
  FORK=$(ls -1 "$OUT/${NAME}_decayed"/snap_*.pt | tail -1)
  DATA=$([ "$NAME" = b ] && echo data/flagship_mix_b || echo data/anneal_mix_c)
  python -m mote.cli train start -- "${COMMON[@]}" "${EXTRAS[@]}" --init-from "$FORK" \
      --data "$DATA" --schedule constant --max-minutes "$TAIL" \
      --out "$OUT/${NAME}_constant"
done

SFT_ARGS="--preset flagship --data data/sft_local --sft --mix data/sim_sft:0.10 \
  --optimizer adamw --lr 3e-4 --batch-size 1 --grad-accum 8 --seq-len 4096 --ckpt-main \
  --max-minutes 60 --eval-every 300 --ckpt-minutes 5"

echo "== 3. the identical 60-min SFT for all five arms"
# The floor is the trunk snapshot with no mid-training at all. It carries none of the extras, so its
# capability numbers are a LOWER BOUND and not a contestant -- it is here to say what the whole stage buys.
for ARM in b_decayed b_constant c_decayed c_constant; do
  python -m mote.cli train start -- $SFT_ARGS --init-from "$OUT/$ARM/last.pt" --out "$OUT/${ARM}_sft"
done
python -m mote.cli train start -- $SFT_ARGS --init-from "$SNAP" --out "$OUT/floor_sft"
python -m mote.cli train queue

echo "== 4. the decider on every arm, then the gate on the two decayed branches"
for ARM in b_decayed b_constant c_decayed c_constant floor; do
  python -m mote.eval.proxy --checkpoint "$OUT/${ARM}_sft/last.pt" --n-sim 120 --n-read 100
done
python -m mote.eval.branch_gate --skip-sft \
    --branch control="$OUT/b_decayed" --branch anneal="$OUT/c_decayed" \
    --sft-run control="$OUT/b_decayed_sft" --sft-run anneal="$OUT/c_decayed_sft" \
    --out "docs/results/$(date +%F)-mid-2x2.md"

echo "== 5. the two axes, read off the same five checkpoints"
python scripts/mid_2x2_report.py
