#!/usr/bin/env bash
# Day one on Fedora: the six measurements in docs/shape.md, written next to their WSL2 twins.
# Untested until it runs there (written on Windows 2026-08-23). Run from the repo root with the venv active:
#     bash scripts/fedora_day1.sh 2>&1 | tee docs/results/$(date +%F)-fedora-day1.log
set -u
OUT=docs/results/$(date +%F)-fedora-day1
mkdir -p "$OUT"
echo "== 0. environment"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
python -c "from mote.model import mamba3, relation; print('kernel', mamba3.HAS_MAMBA3_KERNEL, 'triton', relation.HAS_TRITON, 'flash', relation.USE_FLASH)"

echo "== 1. step profile (WSL2 twins: 80.8 KB/s 23.5 % MFU 2.70 GB; 42.4 KB/s 43.7 % 6.34 GB)"
python -m mote.train.profile_step --data data/local_mix --preset local --init-from runs/overnight/last.pt --batch-size 4 --grad-accum 4 --bucket 64 | tee "$OUT/profile_local_2048_b4.txt"
python -m mote.train.profile_step --data data/local_mix --preset flagship --chunk-bytes 6 --seq-len 16384 --batch-size 1 --ckpt-main | tee "$OUT/profile_flagship_16384_b1.txt"

echo "== 2. where the time and memory go"
if command -v nsys >/dev/null; then
  nsys profile -o "$OUT/nsys_local_2048" --force-overwrite true --stats=true \
    python -m mote.train.profile_step --data data/local_mix --preset local --init-from runs/overnight/last.pt --batch-size 4 --grad-accum 4 --bucket 64 --warmup 2 --timed 20 > "$OUT/nsys_local_2048.txt" 2>&1
  nsys profile -o "$OUT/nsys_flagship_16384" --force-overwrite true --stats=true \
    python -m mote.train.profile_step --data data/local_mix --preset flagship --chunk-bytes 6 --seq-len 16384 --batch-size 1 --ckpt-main --warmup 2 --timed 20 > "$OUT/nsys_flagship_16384.txt" 2>&1
else
  echo "nsys not installed (sudo dnf install nsight-systems from the NVIDIA repo); writing Chrome traces instead"
  python -m mote.train.profile_step --data data/local_mix --preset local --init-from runs/overnight/last.pt --batch-size 4 --grad-accum 4 --bucket 64 --trace "$OUT/trace_local_2048.json" > /dev/null
  python -m mote.train.profile_step --data data/local_mix --preset flagship --chunk-bytes 6 --seq-len 16384 --batch-size 1 --ckpt-main --trace "$OUT/trace_flagship_16384.json" > /dev/null
fi
python - "$OUT" <<'EOF'
# peak-memory snapshot of one flagship step, grouped by allocation site (torch's own recorder)
import sys, torch, pickle, collections
from mote.train import profile_step
out = sys.argv[1]
torch.cuda.memory._record_memory_history(max_entries=200000)
try:
    profile_step.main(["--data", "data/local_mix", "--preset", "flagship", "--chunk-bytes", "6", "--seq-len", "16384",
                       "--batch-size", "1", "--ckpt-main", "--warmup", "1", "--timed", "2"])
finally:
    snap = torch.cuda.memory._snapshot()
    pickle.dump(snap, open(f"{out}/memory_flagship_16384.pickle", "wb"))
    torch.cuda.memory._record_memory_history(enabled=None)
by_site = collections.Counter()
for seg in snap["segments"]:
    for blk in seg.get("blocks", []):
        if blk.get("state") != "active_allocated":
            continue
        frames = blk.get("frames") or []
        site = next((f"{f['filename'].split('/')[-1]}:{f['line']} {f['name']}" for f in frames if "/mote/" in f.get("filename", "").replace("\\", "/")), "other")
        by_site[site] += blk["size"]
print("active allocations at the snapshot, by first mote/ frame:")
for site, n in by_site.most_common(25):
    print(f"{n / 2**20:9.1f} MB  {site}")
EOF

echo "== 3. step time at the flagship: read elapsed_s per step from the flagship profile above"

echo "== 4. serving on the kernels (Windows CPU twins: 2.1 s cold, 127 ms warm, 40 B/s)"
python -m mote.eval.prefix_probe --checkpoint runs/overnight_sft2/last.pt --turns 20 --device cuda --out "$OUT/prefix_probe_cuda.json" | tail -15

echo "== 5. sharing the card today: start a training run, then time replies"
echo "   (manual: 'mote service start' on cuda, a 2048 profile in another shell, then the timing script in docs/results/2026-08-23-prefix-cache.md)"

echo "== 6. disk: memmap read throughput of one shard (WSL2 came through 9P)"
SHARD=data/local_mix.train.bin
if [ -n "$SHARD" ]; then
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
  dd if="$SHARD" of=/dev/null bs=16M 2>&1 | tail -1
fi
echo "== done: $OUT"
