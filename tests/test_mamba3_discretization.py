import math
import torch
import pytest

from sisa_mamba.core.mamba3_ssm import Mamba3SSM, Mamba3Config
from sisa_mamba.core.rotary import apply_block_rotations


def test_block_rotation_orthogonality():
    """Verifies that R(Phi)^T R(Phi) = Identity matrix for block 2x2 rotations."""
    torch.manual_seed(42)
    v = torch.randn(4, 8, 32)  # [B, L, d_s]
    phi = torch.randn(4, 8, 16) # [B, L, d_s/2]

    # R(Phi) v
    rotated = apply_block_rotations(v, phi, transpose=False)
    # R(Phi)^T (R(Phi) v) = v
    unrotated = apply_block_rotations(rotated, phi, transpose=True)

    diff = torch.max(torch.abs(unrotated - v)).item()
    assert diff < 1e-6, f"Orthogonality error {diff} is too large!"


def test_mamba3_siso_forward_backward():
    """Verifies Mamba-3 SISO forward pass, shapes, and gradients."""
    config = Mamba3Config(d_model=128, d_state=32, d_head=32, n_heads=4, mimo_rank=1)
    layer = Mamba3SSM(config)

    x = torch.randn(2, 24, 128, requires_grad=True)
    out, state = layer(x)

    assert out.shape == (2, 24, 128)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.in_proj.weight.grad is not None
    assert layer.b_bias.grad is not None
    assert layer.c_bias.grad is not None


def test_mamba3_mimo_forward_backward():
    """Verifies Mamba-3 MIMO (Rank R=4) matrix-matrix recurrence and gradients."""
    config = Mamba3Config(d_model=128, d_state=32, d_head=32, n_heads=4, mimo_rank=4)
    layer = Mamba3SSM(config)

    x = torch.randn(2, 24, 128, requires_grad=True)
    out, state = layer(x)

    assert out.shape == (2, 24, 128)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.x_mimo_scale.grad is not None
    assert layer.mimo_out_proj.weight.grad is not None
