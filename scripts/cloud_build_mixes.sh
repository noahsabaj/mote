#!/usr/bin/env bash
# Rebuild the two mixes inside a Lightning Studio (CPU is enough; both stream from public HF datasets).
# `mote.data.build_mix` is deterministic over file-ordered streams: the result is byte-identical to the
# local build — check per_source_bytes in *.meta.json and md5sum the val shard against the local copy.
set -euo pipefail
ROOT=${ROOT:-/teamspace/studios/this_studio}
export HF_HOME=$ROOT/.hf
cd "$ROOT/mote"
mkdir -p data
python -m mote.data.build_mix --out data/flagship_mix --list flagship --target-gb 10 --val-mb 128 > "$ROOT/build_flagship.log" 2>&1 &
python -m mote.data.build_mix --out data/local_mix --list pretrain --target-gb 1.5 --val-mb 16 > "$ROOT/build_local.log" 2>&1 &
wait
echo "mixes built"
