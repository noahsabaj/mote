"""FlashRelation (Triton) against the materialized fp32 reference: forward, gates, all gradients.
GPU-only; skipped elsewhere."""

import math

import pytest
import torch

from morpheme.model.flash_relation import HAS_TRITON, flash_relation, relation_reference

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="needs CUDA + Triton")


def _inputs(B, H, TQ, S, D, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda T: (torch.randn(B, H, T, D, device="cuda", generator=g) * 0.7).to(dtype).requires_grad_(True)
    p1 = mk(TQ)
    p2 = mk(S + TQ)
    info = mk(S + TQ)
    lam = torch.tensor(0.5, device="cuda", dtype=torch.float32, requires_grad=True)
    return p1, p2, info, lam


@pytest.mark.parametrize("D", [48, 64, 96])
@pytest.mark.parametrize("TQ,S", [(16, 0), (50, 0), (130, 0), (300, 0), (20, 37), (64, 64)])
def test_fp32_matches_reference_tightly(D, TQ, S):
    p1, p2, info, lam = _inputs(2, 3, TQ, S, D, torch.float32)
    y, g = flash_relation(p1, p2, info, lam, 2.0, q_start=S)
    yr, gr = relation_reference(p1, p2, info, lam, 2.0, q_start=S)
    assert torch.allclose(y, yr, atol=2e-4, rtol=1e-3), (y - yr).abs().max()
    assert torch.allclose(g, gr, atol=2e-4), (g - gr).abs().max()
    dy = torch.randn_like(y)
    grads = torch.autograd.grad((y * dy).sum(), [p1, p2, info, lam])
    grads_r = torch.autograd.grad((yr * dy).sum(), [p1, p2, info, lam])
    for a, b, name in zip(grads, grads_r, ["dp1", "dp2", "dinfo", "dlam"]):
        assert torch.allclose(a, b, atol=3e-4, rtol=2e-3), (name, (a - b).abs().max().item())


@pytest.mark.parametrize("D", [48, 64])
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


def test_first_position_is_pure_self():
    p1, p2, info, lam = _inputs(1, 2, 16, 0, 64, torch.float32)
    y, g = flash_relation(p1, p2, info, lam, 2.0)
    assert torch.allclose(y[:, :, 0], info[:, :, 0], atol=1e-6)
    assert float(g[:, :, 0].abs().max()) == 0.0
