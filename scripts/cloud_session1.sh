#!/bin/bash
# Session 1 of the throughput line (docs/results/2026-08-29-throughput-line-prereg.md), run inside a Lightning
# Job's working copy at .../mote on one L4: the forward-only bisect of the housekeeping commits, the --ckpt-main
# probe, the FlashRelation pair, the eager / --compile twins, then a summary table. Never exits non-zero — a
# failed measurement prints its log tail and the rest still runs (a non-zero exit ends the job before its summary).
set +e
export PYTHONUNBUFFERED=1
OUT=runs/cloud/s1; mkdir -p "$OUT"
BASE="--preset mote-96m --data data/flagship_mix --batch-size 1 --seq-len 16384 --optimizer muon --weight-decay 0.1 --schedule trunk --bound-floor 2048 --seed 42 --eval-every 100000 --eval-batches 1 --ckpt-minutes 999"
FW="$BASE --grad-accum 1 --ckpt-main --lr 0 --max-steps 2 --log-every 1"   # lr 0: last.pt is the init, both steps are forwards
t() { date +%T; }
echo "== s1 start $(t) on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

echo "== 0. forward-only bisect (init + two forwards, lr 0) — base 58e8672 and the five housekeeping commits"
git fetch -q origin 2>&1 | tail -2
for sha in 58e8672 8a89fd3 364f341 d2c7c0a cff1dec bbb3d36; do
  wt=../mote-w$sha
  [ -d "$wt" ] || git worktree add -q --detach "$wt" "$sha" 2>&1 | tail -1
  [ -e "$wt/data" ] || ln -s ../mote/data "$wt/data"
  echo "-- fw $sha $(t)"
  (cd "$wt" && python -m mote.train.train $FW --out "../mote/$OUT/fw/$sha") > "$OUT/fw_$sha.log" 2>&1 || { echo "fw $sha FAILED"; tail -5 "$OUT/fw_$sha.log"; }
  grep -o '"train_bpb": [0-9.]*' "$OUT/fw/$sha/log.jsonl" 2>/dev/null | tr '\n' ' '; echo
done
for sha in 8a89fd3 364f341 d2c7c0a cff1dec bbb3d36; do
  printf "bisect %s vs 58e8672: " "$sha"
  python scripts/bitwise_diff.py "$OUT/fw/58e8672" "$OUT/fw/$sha" 2>&1 | grep -E "^log:|^state:|VERDICT|step [12] train_bpb" | tr '\n' ' '; echo
done

echo "== 1. --ckpt-main probe: 20 steps with and without (peak_gb, B/s)"
for v in on off; do
  flag=""; [ "$v" = on ] && flag="--ckpt-main"
  echo "-- ckpt $v $(t)"
  python -m mote.train.train $BASE --grad-accum 2 $flag --lr 8e-4 --max-steps 20 --log-every 1 --out "$OUT/ckpt/$v" > "$OUT/ckpt_$v.log" 2>&1 || { echo "ckpt $v FAILED"; tail -5 "$OUT/ckpt_$v.log"; }
done

echo "== 2. FlashRelation pair: default one-pass backward (v2) vs MOTE_DETERMINISTIC_RELATION=1 two-pass, 30 min each"
for v in v2 twopass; do
  e=""; [ "$v" = twopass ] && e="MOTE_DETERMINISTIC_RELATION=1"
  echo "-- pair $v $(t)"
  env $e python -m mote.train.train $BASE --grad-accum 2 --ckpt-main --lr 8e-4 --max-minutes 30 --log-every 10 --out "$OUT/pair/$v" > "$OUT/pair_$v.log" 2>&1 || { echo "pair $v FAILED"; tail -5 "$OUT/pair_$v.log"; }
done

echo "== 3. eager vs --compile twins: 500 steps each (--stop-minutes 30 caps the wall clock without touching the schedule)"
for v in eager compile; do
  flag=""; [ "$v" = compile ] && flag="--compile"
  echo "-- twin $v $(t)"
  python -m mote.train.train $BASE --grad-accum 2 --ckpt-main --lr 8e-4 --max-steps 500 --stop-minutes 30 --log-every 10 $flag --out "$OUT/twins/$v" > "$OUT/twin_$v.log" 2>&1 || { echo "twin $v FAILED"; tail -8 "$OUT/twin_$v.log"; }
done

echo "== summary $(t)"
python - <<'PY'
import json, glob, statistics as st
print(f"{'run':16s} {'steps':>5s} {'B/s med':>9s} {'peak_gb':>7s} {'last_bpb':>10s} {'first_step_s':>12s}")
for p in sorted(glob.glob("runs/cloud/s1/*/*/log.jsonl")):
    r = [json.loads(l) for l in open(p) if l.startswith("{")]
    r = [x for x in r if "train_bpb" in x]
    if not r: continue
    n = r[-1]["step"]; lo = 200 if n >= 300 else 5
    bs = [x["bytes_per_sec"] for x in r if x["step"] >= lo] or [x["bytes_per_sec"] for x in r]
    name = "/".join(p.split("/")[3:5])
    print(f"{name:16s} {n:5d} {st.median(bs):9.0f} {r[-1].get('peak_gb', 0):7.2f} {r[-1]['train_bpb']:10.4f} {r[0].get('elapsed_min', 0)*60:12.0f}")
PY
echo "== s1 done $(t)"
exit 0
