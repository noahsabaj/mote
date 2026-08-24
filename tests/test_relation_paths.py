"""The Relation module through the fused kernel and through the materialized reference path give the same
outputs and the same parameter gradients (the model-level exactness gate for kernel revisions; docs/shape.md
"Kernel and compile workstreams"). GPU, or the CPU Triton interpreter with TRITON_INTERPRET=1."""

import os

import pytest
import torch

import mote.model.relation as R
from mote.model.flash_relation import HAS_TRITON
from mote.model.relation import FullRelation

DEV = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not HAS_TRITON or (DEV == "cpu" and os.environ.get("TRITON_INTERPRET") != "1"), reason="needs CUDA + Triton (or TRITON_INTERPRET=1)")


def _run(use_flash: bool, x: torch.Tensor, seed: int, d_model: int, n_heads: int, layer_idx: int):
    torch.manual_seed(seed)
    mod = FullRelation(d_model, n_heads, layer_idx=layer_idx, device=DEV, dtype=torch.float32)
    with torch.no_grad():
        mod.theta.uniform_(-0.3, 0.3)
        mod.lam.fill_(0.4)
    R.USE_FLASH = use_flash
    xin = x.clone().requires_grad_(True)
    out, g = mod(xin, return_gates=True)
    (out * torch.linspace(-1, 1, out.numel(), device=DEV).view_as(out)).sum().backward()
    grads = {n: p.grad.clone() for n, p in mod.named_parameters()}
    return out.detach(), g.detach(), xin.grad.clone(), grads


@pytest.mark.parametrize("d_model,n_heads,T", [(192, 2, 130), (128, 2, 70), (96, 2, 40)])  # heads of 96, 64, 48
@pytest.mark.parametrize("layer_idx", [0, 1])  # both Givens pairings
def test_flash_and_materialized_paths_agree(d_model, n_heads, T, layer_idx):
    old = R.USE_FLASH
    try:
        x = torch.randn(2, T, d_model, device=DEV) * 0.8
        a = _run(True, x, 0, d_model, n_heads, layer_idx)
        b = _run(False, x, 0, d_model, n_heads, layer_idx)
    finally:
        R.USE_FLASH = old
    assert torch.allclose(a[0], b[0], atol=3e-4, rtol=1e-3), (a[0] - b[0]).abs().max()
    assert torch.allclose(a[1], b[1], atol=3e-4), (a[1] - b[1]).abs().max()
    assert torch.allclose(a[2], b[2], atol=5e-4, rtol=2e-3), (a[2] - b[2]).abs().max()
    for n in a[3]:
        scale = b[3][n].abs().max().clamp(min=1e-3)
        assert ((a[3][n] - b[3][n]).abs().max() / scale) < 2e-3, (n, ((a[3][n] - b[3][n]).abs().max() / scale).item())
