#!/usr/bin/env bash
# Run one trainer arm inside a Lightning Job and stop it at a target step.
#   cloud_arm.sh <stop_step> <out_dir> -- <mote.train.train args...>
# Why: `--max-steps` also fixes the trunk schedule's warmup (10 % of the horizon), so arms that must share a
# warmup share --max-steps, and the ones meant to end early stop at --stop-step instead. Until 2026-08-30 this
# script polled log.jsonl and SIGTERMed `$!`; on the L4 studio that was a launcher shim, the trainer ran on as
# an orphan to its full horizon and mote-lr-28p8e-4 billed 12.8 h for a 3-h point. The trainer stops itself now.
set -uo pipefail
STOP=$1; OUT=$2; shift 2; [ "${1:-}" = "--" ] && shift
mkdir -p "$OUT"
python -m mote.train.train --out "$OUT" --stop-step "$STOP" "$@"
echo "cloud_arm: trainer exited with $?"
