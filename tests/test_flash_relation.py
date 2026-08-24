"""FlashRelation (Triton) against the materialized fp32 reference: forward, gates, all gradients, both
backward paths (one-pass atomics and the deterministic two-pass fallback), the conditional-rescale branch.
Runs on the GPU; with TRITON_INTERPRET=1 and no CUDA the fp32 cases run on the CPU interpreter."""

import os

import pytest
import torch

import mote.model.flash_relation as fr
from mote.model.flash_relation import HAS_TRITON, flash_relation, relation_reference

DEV = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not HAS_TRITON or (DEV == "cpu" and os.environ.get("TRITON_INTERPRET") != "1"), reason="needs CUDA + Triton (or TRITON_INTERPRET=1)")
gpu_only = pytest.mark.skipif(DEV != "cuda", reason="bf16 tiles need the GPU")


def _inputs(B, H, TQ, S, D, dtype, seed=0, scale=0.7):
    g = torch.Generator(device=DEV).manual_seed(seed)
    mk = lambda T: (torch.randn(B, H, T, D, device=DEV, generator=g) * scale).to(dtype).requires_grad_(True)
    p1 = mk(TQ)
    p2 = mk(S + TQ)
    info = mk(S + TQ)
    lam = torch.tensor(0.5, device=DEV, dtype=torch.float32, requires_grad=True)
    return p1, p2, info, lam


def _check_fp32(p1, p2, info, lam, S, atol=2e-4, gtol=3e-4, dy=None):
    y, g = flash_relation(p1, p2, info, lam, 2.0, q_start=S)
    yr, gr = relation_reference(p1, p2, info, lam, 2.0, q_start=S)
    assert torch.allclose(y, yr, atol=atol, rtol=1e-3), (y - yr).abs().max()
    assert torch.allclose(g, gr, atol=atol), (g - gr).abs().max()
    if dy is None:
        dy = torch.randn_like(y)
    grads = torch.autograd.grad((y * dy).sum(), [p1, p2, info, lam])
    grads_r = torch.autograd.grad((yr * dy).sum(), [p1, p2, info, lam])
    for a, b, name in zip(grads, grads_r, ["dp1", "dp2", "dinfo", "dlam"]):
        assert torch.allclose(a, b, atol=gtol, rtol=2e-3), (name, (a - b).abs().max().item())
    return grads


@pytest.mark.parametrize("D", [48, 64, 96])
@pytest.mark.parametrize("TQ,S", [(16, 0), (50, 0), (130, 0), (300, 0), (20, 37), (64, 64)])
def test_fp32_matches_reference_tightly(D, TQ, S):
    _check_fp32(*_inputs(2, 3, TQ, S, D, torch.float32), S)


@pytest.mark.parametrize("D", [96, 112, 24])
def test_odd_head_dims(D):
    """96 = 64+32 exact; 112 = 64 + a masked 64 block; 24 = one masked 32 block."""
    _check_fp32(*_inputs(1, 2, 90, 0, D, torch.float32), 0)


def test_rescale_branch_large_scores():
    """Scores spanning far more than log 256 across key blocks: the conditional rescale must fire and the
    final normalization must still be exact."""
    _check_fp32(*_inputs(1, 2, 200, 0, 64, torch.float32, seed=3, scale=3.0), 0, atol=5e-4, gtol=2e-3)


def test_two_pass_fallback_matches_one_pass(monkeypatch):
    p1, p2, info, lam = _inputs(2, 2, 150, 10, 96, torch.float32, seed=1)
    dy = torch.randn(2, 2, 150, 96, device=DEV, generator=torch.Generator(device=DEV).manual_seed(7))
    monkeypatch.setattr(fr, "DETERMINISTIC", False)
    fast = _check_fp32(p1, p2, info, lam, 10, dy=dy)
    monkeypatch.setattr(fr, "DETERMINISTIC", True)
    slow = _check_fp32(p1, p2, info, lam, 10, dy=dy)
    for a, b in zip(fast, slow):
        assert torch.allclose(a, b, atol=3e-4, rtol=2e-3)


@gpu_only
@pytest.mark.parametrize("D", [48, 64, 96])
@pytest.mark.parametrize("TQ", [33, 257])
def test_bf16_matches_reference_loosely(D, TQ):
    p1, p2, info, lam = _inputs(2, 4, TQ, 0, D, torch.bfloat16)
    y, g = flash_relation(p1, p2, info, lam, 2.0)
    yr, gr = relation_reference(p1, p2, info, lam, 2.0)
    assert (y.float() - yr).abs().max() < 4e-2
    assert (g - gr).abs().max() < 2e-2
    dy = torch.randn_like(y)
    grads = torch.autograd.grad((y.float() * dy.float()).sum(), [p1, p2, info, lam])
    grads_r = torch.autograd.grad((yr * dy.float()).sum(), [p1, p2, info, lam])
    for a, b, name in zip(grads, grads_r, ["dp1", "dp2", "dinfo", "dlam"]):
        scale = b.abs().max().clamp(min=1e-3)
        assert ((a.float() - b).abs().max() / scale) < 6e-2, (name, ((a.float() - b).abs().max() / scale).item())


@gpu_only
def test_bf16_prefix_and_flagship_shape():
    """The flagship head (D=96) with a cached prefix, bf16, at a chunk count past the bucket size."""
    p1, p2, info, lam = _inputs(1, 8, 700, 64, 96, torch.bfloat16, seed=2)
    y, g = flash_relation(p1, p2, info, lam, 2.0, q_start=64)
    yr, gr = relation_reference(p1, p2, info, lam, 2.0, q_start=64)
    assert (y.float() - yr).abs().max() < 4e-2
    dy = torch.randn_like(y)
    grads = torch.autograd.grad((y.float() * dy.float()).sum(), [p1, p2, info, lam])
    grads_r = torch.autograd.grad((yr * dy.float()).sum(), [p1, p2, info, lam])
    for a, b, name in zip(grads, grads_r, ["dp1", "dp2", "dinfo", "dlam"]):
        scale = b.abs().max().clamp(min=1e-3)
        assert ((a.float() - b).abs().max() / scale) < 6e-2, (name, ((a.float() - b).abs().max() / scale).item())


def test_first_position_is_pure_self():
    p1, p2, info, lam = _inputs(1, 2, 16, 0, 64, torch.float32)
    y, g = flash_relation(p1, p2, info, lam, 2.0)
    assert torch.allclose(y[:, :, 0], info[:, :, 0], atol=1e-6)
    assert float(g[:, :, 0].abs().max()) == 0.0
