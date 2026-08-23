#!/usr/bin/env bash
# Studio-side environment setup for the flagship run (Lightning.ai, H100, Ubuntu).
# Idempotent: safe to re-run after an interruptible preemption.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d .venv-cloud ] || uv venv .venv-cloud --python 3.12
PY=.venv-cloud/bin/python
uv pip install --python $PY -q torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python $PY -q "triton>=3.5" einops numpy huggingface_hub transformers datasets fastapi "uvicorn[standard]" websockets pytest packaging ninja setuptools wheel
# official Mamba-3 kernels, pinned to the commit validated locally (state-spaces/mamba e9594ce)
if ! $PY -c "import mamba_ssm.ops.triton.mamba3.mamba3_siso_combined" 2>/dev/null; then
  [ -d /tmp/mamba ] || git clone -q https://github.com/state-spaces/mamba.git /tmp/mamba
  (cd /tmp/mamba && git checkout -q e9594ce)
  MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --python $PY -q -e /tmp/mamba --no-deps --no-build-isolation
fi
$PY -c "import torch, triton; print('torch', torch.__version__, torch.cuda.get_device_name(0), '| triton', triton.__version__)"
$PY -c "from mote.model.mamba3 import HAS_MAMBA3_KERNEL; print('mamba3 kernel:', HAS_MAMBA3_KERNEL)"
$PY -m pytest -q -x -p no:cacheprovider tests/test_mote_model.py
echo "bootstrap ok"
