#!/usr/bin/env bash
# Set up a Lightning Studio (the free CPU machine is enough) to run Mote arms as Jobs on GPU machines.
#   bash scripts/cloud_setup.sh [commit]          # inside the studio; idempotent
# The env matches the local one: torch 2.13.0+cu126 (pip's default pick is cu130 — wrong for the L4/H100
# images), triton 3.7.1, Mamba-3 kernels from state-spaces/mamba@e9594ce (Triton only, no CUDA build).
# The mixes are never uploaded: `mote.data.build_mix` is deterministic over file-ordered HF streams, so a
# rebuild in the studio is byte-identical to the local one (compare *.meta.json per_source_bytes).
set -euo pipefail
ROOT=${ROOT:-/teamspace/studios/this_studio}
COMMIT=${1:-main}
MAMBA_COMMIT=e9594ce
export HF_HOME=$ROOT/.hf   # the studio's default cache carries a stale token -> 401 on public datasets

cd "$ROOT"
[ -d mote ] || git clone -q https://github.com/noahsabaj/mote
git -C mote fetch -q origin && git -C mote checkout -q "$COMMIT"
pip install -q "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu126
pip install -q "triton==3.7.1" einops transformers   # mamba_ssm imports transformers at package import
pip install -q -e mote
[ -d mamba ] || git clone -q https://github.com/state-spaces/mamba
git -C mamba checkout -q "$MAMBA_COMMIT"
(cd mamba && MAMBA_SKIP_CUDA_BUILD=TRUE pip install -q -e . --no-deps --no-build-isolation)
python - <<'PY'
import torch, triton, mamba_ssm, mote
print("torch", torch.__version__, "| triton", triton.__version__, "| cuda available", torch.cuda.is_available())
PY
echo "setup ok: mote@$(git -C mote rev-parse --short HEAD) mamba@$MAMBA_COMMIT HF_HOME=$HF_HOME"
