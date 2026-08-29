#!/usr/bin/env bash
# Run one trainer arm inside a Lightning Job and stop it at a target step.
#   cloud_arm.sh <stop_step> <out_dir> -- <mote.train.train args...>
# Why: `--max-steps` also fixes the trunk schedule's warmup (10 % of the horizon), so arms that must share a
# warmup share --max-steps, and the ones meant to end early are stopped here instead. log.jsonl is flushed
# per record, so a SIGTERM loses nothing the read needs.
set -uo pipefail
STOP=$1; OUT=$2; shift 2; [ "${1:-}" = "--" ] && shift
mkdir -p "$OUT"
python -m mote.train.train --out "$OUT" "$@" &
PID=$!
N=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
  STEP=$(python -c '
import json, sys
step = 0
try:
    for line in open(sys.argv[1]):
        try: step = max(step, int(json.loads(line).get("step", 0)))
        except Exception: pass
except FileNotFoundError: pass
print(step)' "$OUT/log.jsonl")
  if [ "$STEP" -ge "$STOP" ]; then echo "cloud_arm: step $STEP >= $STOP, sending SIGTERM"; kill -TERM "$PID"; break; fi
  N=$((N + 1)); if [ $((N % 10)) -eq 0 ]; then echo "cloud_arm: $(date -u +%H:%M) step $STEP | $(tail -1 "$OUT/log.jsonl" 2>/dev/null | cut -c1-200)"; fi
done
wait "$PID"; echo "cloud_arm: trainer exited with $?"
