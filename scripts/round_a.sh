#!/usr/bin/env bash
# Round A — which preference objective, and does the fixed data stop the false firing?
#
# Signed 2026-08-25 (docs/shape.md § post, docs/research/dpo-rlvr-2026-08-25.md). Diagnostic, not the
# decision: 1000 identity pairs where the failure is characterised and the probe measures it directly.
# Round B then runs the winner and runner-up on the 20k sim pairs against the real gate.
#
# Seven arms over two axes that were confounded until 2026-08-25:
#   objective          dpo | ipo | orpo
#   length-normalised  no | yes   (reply length predicted the label in 400/400 pushback pairs; the data is
#                                  fixed now, so this measures whether normalisation still buys anything)
# plus two SFT mixes, because 70% of the false firing is in the SFT checkpoint before DPO ever runs.
# ORPO replaces SFT rather than following it, so it starts from the base and has no mix axis.
#
# The card must be free — this queues nothing. ~95 GPU-minutes: 2 x 30 min SFT + 7 x ~3 min + 9 probes.
# Run from the repo root with the venv active, after the training queue drains:
#     bash scripts/round_a.sh 2>&1 | tee docs/results/$(date +%F)-round-a.log
set -u
BASE=runs/t3l_dense_4e-4/last.pt          # 31.6M, val_bpb_ema 1.0276 — the best of the three 12-h LR arms
                                          # (8e-4 1.0370, 16e-4 1.0774; swapped 2026-08-26 when the sweep finished)
OUT=runs/roundA
PARAMS=31643528
mkdir -p "$OUT" docs/results

echo "== 0. data: pushback + swap + negative class + ties, and the two SFT mixes"
# The pairs are identical in both builds (same seed, same generator); only the SFT dialogue mix differs.
python -m mote.data.build_identity --out data/ra_id_plain --params $PARAMS \
    --n 4000 --pairs 400 --swap 200 --neg 200 --ties 200 --neutral-frac 0
python -m mote.data.build_identity --out data/ra_id_neutral --params $PARAMS \
    --n 4000 --pairs 400 --swap 200 --neg 200 --ties 200 --neutral-frac 0.15
cp data/ra_id_plain.dpo.jsonl "$OUT/pairs.jsonl"

echo "== 1. the two shared SFT checkpoints (30 min each)"
for MIX in plain neutral; do
  python -m mote.train.train --preset local --init-from "$BASE" --sft --data "data/ra_id_${MIX}" \
      --out "$OUT/sft_$MIX" --optimizer muon --lr 3e-4 --batch-size 4 --grad-accum 4 --seq-len 2048 \
      --max-minutes 30 --eval-every 500 --ckpt-minutes 10 --no-mbp --eval-spread
  python -m mote.eval.probe --checkpoint "$OUT/sft_$MIX/last.pt" --out "$OUT/sft_$MIX/probe.json" | tail -3
done

echo "== 2. the preference arms (~3 min each)"
for MIX in plain neutral; do
  for OBJ in dpo ipo; do
    for LN in "" "--length-norm"; do
      TAG="${OBJ}_${MIX}$([ -n "$LN" ] && echo _ln)"
      python -m mote.train.dpo --init-from "$OUT/sft_$MIX/last.pt" --pairs "$OUT/pairs.jsonl" \
          --out "$OUT/$TAG" --objective "$OBJ" $LN --epochs 1 --lr 5e-7 --beta 0.1 --sft-weight 0.5
      python -m mote.eval.probe --checkpoint "$OUT/$TAG/last.pt" --out "$OUT/$TAG/probe.json" | tail -3
    done
  done
done
# ORPO is reference-free and single-stage: it replaces SFT, so it starts from the base.
python -m mote.train.dpo --init-from "$BASE" --pairs "$OUT/pairs.jsonl" \
    --out "$OUT/orpo" --objective orpo --epochs 1 --lr 5e-7 --orpo-lambda 1.0
python -m mote.eval.probe --checkpoint "$OUT/orpo/last.pt" --out "$OUT/orpo/probe.json" | tail -3

echo "== 3. the table. false_fire_rate FIRST: identity_acc cannot tell 'knows what it is' from 'says it"
echo "     to everything' — overnight_dpo2 scored 0.833 on it while false-firing 9 times in 10."
python - <<'PY'
import json, pathlib
rows = []
for d in sorted(pathlib.Path("runs/roundA").glob("*/probe.json")):
    p = json.loads(d.read_text())
    rows.append((d.parent.name, p["false_fire_rate"], p["identity_recite_rate"], p["template_fire_rate"],
                 p["identity_acc"], p["hold_rate"], p["concede_rate"], p.get("n_neutral", 0)))
print(f"{'arm':18s} {'false_fire':>10s} {'recite':>7s} {'template':>9s} {'ident':>6s} {'hold':>6s} {'concede':>8s} {'n':>4s}")
for r in sorted(rows, key=lambda r: r[1]):
    print(f"{r[0]:18s} {r[1]:10.3f} {r[2]:7.3f} {r[3]:9.3f} {r[4]:6.3f} {r[5]:6.3f} {r[6]:8.3f} {r[7]:4d}")
print("\nbaselines measured 2026-08-25 on the old stack: overnight_sft2 0.70, overnight_dpo2 0.90 (n=10)")
print("with n_neutral=40 one prompt is 2.5 points, so read gaps of <5 points as noise.")
PY
