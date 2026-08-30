#!/usr/bin/env bash
# The Monday gap (docs/results/2026-08-28-lr-prereg.md § Decision; docs/results/2026-08-29-housekeeping-prereg.md
# amendment 2026-08-29): everything that must be read before the trunk launches, on the local card, after the queue
# drains. Run from the main checkout with the daemon up and its queue idle; the script parks the studio's engine on
# the CPU first and restores it at the end. Never exits non-zero — a failed measurement prints its log tail and the
# rest still runs; the summary at the end is what the launch decision reads. Results under runs/gap/.
#
#   scripts/gap_protocol.sh                  # all phases
#   PHASES="tests envelope" scripts/gap_protocol.sh
#
# Phases: tests    — the GPU-only pytest files, one at a time
#         bisect   — forward-only bisect of the housekeeping commits at the flagship shape (2 forwards, lr 0)
#         envelope — 3 HEAD × 100 steps + 1 old-code × 100 steps at the trunk argv; scripts/envelope.py
#         s1       — the throughput line's session 1: --ckpt-main probe, FlashRelation pair, eager/--compile twins
#         serve    — fixed-seed serving diff old vs new engine on the CPU (raw weights) + the boundary check
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd); PY=$ROOT/.venv/bin/python; OUT=$ROOT/runs/gap; mkdir -p "$OUT"
PHASES=${PHASES:-"tests bisect envelope s1 serve"}
OLD=58e8672; HK="8a89fd3 364f341 d2c7c0a cff1dec bbb3d36"
TRUNK="--preset mote-96m --data data/flagship_mix --batch-size 1 --seq-len 16384 --optimizer muon --weight-decay 0.1 --schedule trunk --bound-floor 2048 --seed 42 --eval-every 100000 --eval-batches 1 --ckpt-minutes 999"
FW="$TRUNK --grad-accum 1 --ckpt-main --lr 0 --max-steps 2 --log-every 1"
t() { date +%T; }
has() { [[ " $PHASES " == *" $1 "* ]]; }
worktree() {  # a detached checkout of an old commit beside the repo, with the data linked in
  local sha=$1 wt=$ROOT/../mote-w$sha
  [ -d "$wt" ] || git worktree add -q --detach "$wt" "$sha" 2>&1 | tail -1
  [ -e "$wt/data" ] || ln -s "$ROOT/data" "$wt/data"
  echo "$wt"
}
echo "== gap protocol start $(t) at $(git rev-parse --short HEAD) on $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader)"
$PY -m mote.cli engine park | tr '\n' ' '; echo

if has tests; then
  echo "== tests $(t): the GPU-only files, one at a time"
  for f in test_graph_decode test_flash_relation test_determinism test_fused_norm test_mamba3_states test_relation_paths test_moe; do
    $PY -m pytest -x -q -p no:cacheprovider tests/$f.py > "$OUT/pytest_$f.log" 2>&1
    echo "  $f exit=$? $(tail -1 "$OUT/pytest_$f.log")"
  done
fi

if has bisect; then
  echo "== bisect $(t): forward-only, init + two forwards at lr 0, $OLD then the five housekeeping commits, then HEAD"
  for sha in $OLD $HK; do
    wt=$(worktree $sha)
    (cd "$wt" && PYTHONPATH="$wt" $PY -m mote.train.train $FW --out "$OUT/fw/$sha") > "$OUT/fw_$sha.log" 2>&1 || { echo "  fw $sha FAILED"; tail -4 "$OUT/fw_$sha.log"; }
  done
  $PY -m mote.train.train $FW --out "$OUT/fw/head" > "$OUT/fw_head.log" 2>&1 || { echo "  fw head FAILED"; tail -4 "$OUT/fw_head.log"; }
  for sha in $HK head; do
    printf "  %s vs %s: " "$sha" "$OLD"
    $PY scripts/bitwise_diff.py "$OUT/fw/$OLD" "$OUT/fw/$sha" 2>&1 | grep -E "^log:|VERDICT|step [12] train_bpb" | tr '\n' ' '; echo
  done
fi

if has envelope; then
  echo "== envelope $(t): 3 × HEAD and 1 × $OLD, 100 steps at the trunk argv, lr 1.0e-4"
  ENV="$TRUNK --grad-accum 2 --ckpt-main --lr 1.0e-4 --max-steps 100 --log-every 1"
  for n in a b c; do
    $PY -m mote.train.train $ENV --out "$OUT/env/$n" > "$OUT/env_$n.log" 2>&1 || { echo "  env $n FAILED"; tail -4 "$OUT/env_$n.log"; }
  done
  wt=$(worktree $OLD)
  (cd "$wt" && PYTHONPATH="$wt" $PY -m mote.train.train $ENV --out "$OUT/env/old") > "$OUT/env_old.log" 2>&1 || { echo "  env old FAILED"; tail -4 "$OUT/env_old.log"; }
  $PY scripts/envelope.py "$OUT/env" a b c --old old 2>&1 | tail -30
fi

if has s1; then
  echo "== s1 $(t): the throughput line's session 1 on the local card (docs/results/2026-08-29-throughput-line-prereg.md)"
  echo "-- 1. --ckpt-main probe: 20 steps with and without (peak_gb, B/s; an OOM without it is the answer on 8 GB)"
  for v in on off; do
    flag=""; [ "$v" = on ] && flag="--ckpt-main"
    $PY -m mote.train.train $TRUNK --grad-accum 2 $flag --lr 1.0e-4 --max-steps 20 --log-every 1 --out "$OUT/ckpt/$v" > "$OUT/ckpt_$v.log" 2>&1 || { echo "  ckpt $v FAILED: $(grep -m1 -o 'OutOfMemoryError.*' "$OUT/ckpt_$v.log" | cut -c1-80)"; }
  done
  echo "-- 2. FlashRelation pair $(t): default one-pass backward vs MOTE_DETERMINISTIC_RELATION=1 two-pass, 30 min each"
  for v in v2 twopass; do
    e=""; [ "$v" = twopass ] && e="MOTE_DETERMINISTIC_RELATION=1"
    env $e $PY -m mote.train.train $TRUNK --grad-accum 2 --ckpt-main --lr 1.0e-4 --max-steps 61035 --stop-minutes 30 --log-every 10 --out "$OUT/pair/$v" > "$OUT/pair_$v.log" 2>&1 || { echo "  pair $v FAILED"; tail -4 "$OUT/pair_$v.log"; }
  done
  echo "-- 3. eager vs --compile twins $(t): 500 steps each, --stop-minutes 30"
  for v in eager compile; do
    flag=""; [ "$v" = compile ] && flag="--compile"
    $PY -m mote.train.train $TRUNK --grad-accum 2 --ckpt-main --lr 1.0e-4 --max-steps 61035 --stop-step 500 --stop-minutes 30 --log-every 10 $flag --out "$OUT/twins/$v" > "$OUT/twin_$v.log" 2>&1 || { echo "  twin $v FAILED"; tail -6 "$OUT/twin_$v.log"; }
  done
fi

if has serve; then
  echo "== serve $(t): fixed-seed serving diff on the CPU, raw weights under both engines, + the boundary check"
  CK=${SERVE_CKPT:-runs/t3l_dense_8e-4/last.pt}
  wt=$(worktree $OLD)
  (cd "$wt" && PYTHONPATH="$wt" $PY "$ROOT/scripts/serve_diff.py" "$ROOT/$CK" "$OUT/serve_old.json" --raw) > "$OUT/serve_old.log" 2>&1 || echo "  serve old: exit $? (see $OUT/serve_old.log)"
  $PY scripts/serve_diff.py "$CK" "$OUT/serve_new.json" --raw > "$OUT/serve_new.log" 2>&1 || echo "  serve new: exit $? (see $OUT/serve_new.log)"
  grep -h "boundary check" "$OUT/serve_new.log" | head -4
  $PY scripts/serve_diff.py --compare "$OUT/serve_old.json" "$OUT/serve_new.json" 2>&1 | tail -6
  echo "-- the EMA the daemon serves (HEAD, no --raw): boundary check only"
  $PY scripts/serve_diff.py "$CK" "$OUT/serve_ema.json" > "$OUT/serve_ema.log" 2>&1; grep -h "boundary check\|weights:" "$OUT/serve_ema.log" | head -5
fi

echo "== summary $(t)"
$PY - <<'PY'
import glob, json, statistics as st
print(f"{'run':18s} {'steps':>5s} {'B/s med':>9s} {'peak_gb':>7s} {'last_bpb':>10s} {'min':>6s}")
for p in sorted(glob.glob("runs/gap/*/*/log.jsonl")):
    r = [json.loads(l) for l in open(p) if l.startswith("{")]
    r = [x for x in r if "train_bpb" in x]
    if not r: continue
    n = r[-1]["step"]; lo = 200 if n >= 300 else max(2, n // 4)
    bs = [x["bytes_per_sec"] for x in r if x["step"] >= lo and "bytes_per_sec" in x] or [x.get("bytes_per_sec", 0) for x in r]
    print(f"{'/'.join(p.split('/')[2:4]):18s} {n:5d} {st.median(bs):9.0f} {r[-1].get('peak_gb', 0):7.2f} {r[-1]['train_bpb']:10.4f} {r[-1].get('elapsed_min', 0):6.1f}")
PY
$PY -m mote.cli engine restore | tr '\n' ' '; echo
echo "== gap protocol done $(t)"
