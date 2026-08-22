import math
import torch
import torch.nn.functional as F
import pytest

from sisa_mamba.core.sisa_attention import SISAAttention, SISAConfig
from sisa_mamba.core.rotary import apply_block_rotations


def test_sisa_proposition_1_algebraic_equivalence():
    """
    Directly tests Proposition 1 from SISA paper:
    Let s = d_h^(1/4) * sqrt(lambda), Q_hat_i = [q_i; s C_bar_i], K_hat_j = [k_j; s B_bar_j].
    Then (Q_hat_i^T K_hat_j) / sqrt(d_h) = (q_i^T k_j) / sqrt(d_h) + lambda * C_bar_i^T B_bar_j.
    """
    torch.manual_seed(42)
    B, H, L = 2, 4, 16
    d_h = 64
    d_s = 32
    lam_val = 0.45

    q = torch.randn(B, H, L, d_h, dtype=torch.float64)
    k = torch.randn(B, H, L, d_h, dtype=torch.float64)
    c_bar = torch.randn(B, H, L, d_s, dtype=torch.float64)
    b_bar = torch.randn(B, H, L, d_s, dtype=torch.float64)

    # 1. Manual Score Formula
    content_score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_h)  # [B, H, L, L]
    ssm_score = lam_val * torch.matmul(c_bar, b_bar.transpose(-1, -2))     # [B, H, L, L]
    expected_score = content_score + ssm_score

    # 2. Augmented Q/K with scale = 1 / sqrt(d_h)
    s = (d_h ** 0.25) * math.sqrt(lam_val)
    q_aug = torch.cat([q, s * c_bar], dim=-1)  # [B, H, L, d_h + d_s]
    k_aug = torch.cat([k, s * b_bar], dim=-1)  # [B, H, L, d_h + d_s]

    sdpa_logits = torch.matmul(q_aug, k_aug.transpose(-1, -2)) / math.sqrt(d_h)

    diff = torch.max(torch.abs(sdpa_logits - expected_score)).item()
    assert diff < 1e-10, f"Max difference {diff} exceeds tolerance 1e-10!"


def test_sisa_attention_forward_and_backward():
    """Verifies that SISAAttention runs forward and backward smoothly and updates lambda."""
    config = SISAConfig(d_model=128, n_heads=4, d_head=32, d_s=16, max_seq_len=64)
    layer = SISAAttention(config)

    x = torch.randn(2, 32, 128, requires_grad=True)
    out, cache = layer(x)

    assert out.shape == (2, 32, 128)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.lambda_raw.grad is not None
    assert layer.w_alpha.weight.grad is not None
