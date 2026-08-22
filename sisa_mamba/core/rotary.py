import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


def compute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes standard positional RoPE cosine and sine tables.
    dim: head dimension (must be even)
    max_seq_len: maximum sequence length
    """
    assert dim % 2 == 0, f"Dimension {dim} must be even for RoPE."
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # [seq_len, dim/2]
    # Expand to match [seq_len, dim]
    cos = freqs.cos().repeat_interleave(2, dim=-1)  # [seq_len, dim]
    sin = freqs.sin().repeat_interleave(2, dim=-1)  # [seq_len, dim]
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies standard RoPE to tensor x of shape [..., seq_len, dim].
    cos, sin have shape [seq_len, dim] or broadcastable.
    """
    # x shape: [batch, heads, seq_len, dim] or [batch, seq_len, heads, dim]
    # Rotate half
    d = x.shape[-1]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated_x = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (x * cos) + (rotated_x * sin)


def apply_block_rotations(v: torch.Tensor, phi: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    """
    Applies block 2x2 rotation matrices R(Phi) or R(Phi)^T to vector v.
    v: tensor of shape [..., d_s] where d_s is even.
    phi: tensor of rotation angles of shape [..., d_s / 2] (per-dimension angle).
    transpose: if True, applies R(Phi)^T = R(-Phi).
    
    For 2x2 rotation block:
    [ cos(theta)  -sin(theta) ] [ v_0 ] = [ v_0 cos(theta) - v_1 sin(theta) ]
    [ sin(theta)   cos(theta) ] [ v_1 ]   [ v_0 sin(theta) + v_1 cos(theta) ]
    
    If transposed (transpose=True):
    [ cos(theta)   sin(theta) ] [ v_0 ] = [ v_0 cos(theta) + v_1 sin(theta) ]
    [-sin(theta)   cos(theta) ] [ v_1 ]   [-v_0 sin(theta) + v_1 cos(theta) ]
    """
    d_s = v.shape[-1]
    assert d_s % 2 == 0, f"Vector dimension {d_s} must be even."
    assert phi.shape[-1] == d_s // 2, f"Angle dimension {phi.shape[-1]} must match d_s/2 = {d_s // 2}."

    cos = torch.cos(phi)  # [..., d_s / 2]
    sin = torch.sin(phi)  # [..., d_s / 2]

    v0 = v[..., 0::2]  # [..., d_s / 2]
    v1 = v[..., 1::2]  # [..., d_s / 2]

    if not transpose:
        out0 = v0 * cos - v1 * sin
        out1 = v0 * sin + v1 * cos
    else:
        out0 = v0 * cos + v1 * sin
        out1 = -v0 * sin + v1 * cos

    # Interleave back: out is [..., d_s]
    out = torch.stack((out0, out1), dim=-1).flatten(-2)
    return out
