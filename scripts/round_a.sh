#!/usr/bin/env bash
# Round A — which preference objective, and does the negative class fix the false firing?
#
# Signed 2026-08-25 (docs/shape.md § post, docs/research/dpo-rlvr-2026-08-25.md). Diagnostic, not the
# decision: 400 identity pairs where the failure is already characterised and the probe measures it
# directly. Round B then runs the winner and runner-up on the 20k sim pairs against the real gate.
#
# The card must be free — this queues nothing, it runs in the foreground. ~75 GPU-minutes:
#   2 x 30 min SFT (the --neutral-frac arms)  +  5 x ~3 min preference runs  +  6 probe passes.
# Run from the repo root with the venv active, after the training queue drains:
#     bash scripts/round_a.sh 2>&1 | tee docs/results/$(date +%F)-round-a.log
set -u
BASE=runs/t3l_dense_8e-4/last.pt          # 31.6M, val_bpb 1.1265 — the best-trained current-arch checkpoint
OUT=runs/roundA
PARAMS=31643528
mkdir -p "$OUT" docs/results

echo "== 0. data: pushback + negative class + coin-flipped ties, and the two SFT mixes"
python -m mote.data.build_identity --out data/ra_id_plain --params $PARAMS \
    --n 4000 --pairs 400 --neg 200 --ties 200 --neutral-frac 0
python -m mote.data.build_identity --out data/ra_id_neutral --params $PARAMS \
    --n 4000 --pairs 400 --neg 200 --ties 200 --neutral-frac 0.15
# the pairs are identical in both (same seed, same generator); the SFT mix is the only difference
cp data/ra_id_plain.dpo.jsonl "$OUT/pairs.jsonl"

echo "== 1. the two shared SFT checkpoints (30 min each)"
for MIX in plain neutral; do
  python -m mote.train.train --preset local --init-from "$BASE" --sft --data "data/ra_id_${MIX}" \
      --out "$OUT/sft_$MIX" --optimizer muon --lr 3e-4 --batch-size 4 --grad-accum 4 --seq-len 2048 \
      --max-minutes 30 --eval-every 500 --ckpt-minutes 10 --no-mbp --eval-spread
  python -m mote.eval.probe --checkpoint "$OUT/sft_$MIX/last.pt" --out "$OUT/sft_$MIX/probe.json" | tail -20
done

echo "== 2. the preference arms (~3 min each). DPO and IPO follow SFT; ORPO replaces it, so it starts"
echo "     from the base — base -> ORPO against base -> SFT -> {DPO, IPO}, not three sibling arms."
for MIX in plain neutral; do
  for OBJ in dpo ipo; do
    python -m mote.train.dpo --init-from "$OUT/sft_$MIX/last.pt" --pairs "$OUT/pairs.jsonl" \
        --out "$OUT/${OBJ}_$MIX" --objective "$OBJ" --epochs 1 --lr 5e-7 --beta 0.1 --sft-weight 0.5
    python -m mote.eval.probe --checkpoint "$OUT/${OBJ}_$MIX/last.pt" --out "$OUT/${OBJ}_$MIX/probe.json" | tail -20
  done
done
python -m mote.train.dpo --init-from "$BASE" --pairs "$OUT/pairs.jsonl" \
    --out "$OUT/orpo" --objective orpo --epochs 1 --lr 5e-7 --orpo-lambda 1.0
python -m mote.eval.probe --checkpoint "$OUT/orpo/last.pt" --out "$OUT/orpo/probe.json" | tail -20

echo "== 3. the table. false_fire_rate FIRST: identity_acc alone cannot tell 'knows what it is' from"
echo "     'says it to everything' — overnight_dpo2 scored 0.833 on it while false-firing 9 times in 10."
python - <<'PY'
import json, pathlib
rows = []
for d in sorted(pathlib.Path("runs/roundA").glob("*/probe.json")):
    p = json.loads(d.read_text())
    rows.append((d.parent.name, p["false_fire_rate"], p["identity_recite_rate"], p["template_fire_rate"],
                 p["identity_acc"], p["hold_rate"], p["concede_rate"]))
print(f"{'arm':16s} {'false_fire':>10s} {'recite':>7s} {'template':>9s} {'ident':>6s} {'hold':>6s} {'concede':>8s}")
for r in sorted(rows, key=lambda r: r[1]):
    print(f"{r[0]:16s} {r[1]:10.3f} {r[2]:7.3f} {r[3]:9.3f} {r[4]:6.3f} {r[5]:6.3f} {r[6]:8.3f}")
print("\nbaselines measured 2026-08-25 on the old stack: overnight_sft2 0.70, overnight_dpo2 0.90")
PY
