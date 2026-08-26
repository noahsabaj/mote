#!/usr/bin/env bash
# Choose --p-fail by measurement, not by taste (signed 2026-08-26).
#
# Until 2026-08-26 the sim scripted only legal actions, so p_fail was 0 and every tool result was a
# successful restatement of its call. Any rate above 0 is a change to the world the questions are asked
# about, so the rate is picked by what it does to the sim probe rather than by which number sounds right:
# too low and the model gets no signal that refusals exist, too high and the narratives stop being
# trackable at all.
#
# Read `em` (does the world stay solvable) together with `recovery` on the same checkpoints. A rate that
# holds EM inside its noise band while moving recovery is the one to take.
#
#   bash scripts/sweep_p_fail.sh runs/t3l_dense_4e-4/last.pt 2>&1 | tee docs/results/$(date +%F)-p-fail.log
set -euo pipefail
CKPT="${1:?usage: sweep_p_fail.sh runs/<arm>/last.pt}"
OUT="${OUT:-/tmp/pfail}"
mkdir -p "$OUT" docs/results

for PF in 0 5 15 30; do
  echo "== p_fail=${PF}% =="
  python -m mote.sim.generate --out "$OUT/sim_$PF" --mb 8 --dpo-pairs 200 --p-fail "$PF" >/dev/null
  python - "$OUT/sim_$PF" "$PF" <<'PY'
import json, sys, collections
prefix, pf = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f"{prefix}.narrative.jsonl")]
nb = [len(r["text"].encode()) for r in rows]
fails = sum(1 for r in rows if "tried to" in r["text"] or "попыта" in r["text"] or "としたが" in r["text"])
print(f"   docs {len(rows):5d}  mean bytes {sum(nb)//len(nb):5d}  with a refusal {fails/len(rows):5.0%}")
PY
done

echo
echo "== the probe, on the same checkpoint, against each regenerated set =="
echo "   (sim_probe reads held-out worlds from the CURRENT generator, so run it after each rebuild)"
python -m mote.eval.sim_probe --checkpoint "$CKPT" --n 120 --out "$OUT/probe.json" | tail -3
python -m mote.eval.recovery_probe --checkpoint "$CKPT" --n 60 --out "$OUT/recovery.json" | tail -2
echo
echo "Take the largest rate whose sim-QA EM is still within noise of p_fail=0, and prefer it over a"
echo "smaller one if recovery_rate moves. Record the choice in docs/results and set P_FAIL for mid_2x2.sh."
