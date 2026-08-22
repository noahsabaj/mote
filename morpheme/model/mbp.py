"""Multi-byte prediction head with Latent Causal Attention (Owodunni et al. 2026), adapted to H-Net.

For the byte at position t inside chunk k (offset m = t - start_k) the head's input is
    x_t = z_k + W_r x̂_{start_k} + pos(m)
where z_k is the main network's (dechunked) output for chunk k and x̂_{start_k} the encoder state
at the chunk's first byte — everything depends only on bytes <= start_k, so all bytes of a chunk
can be predicted in parallel. The LCA mask lets position t attend to earlier positions of its own
chunk and to every position of the previous chunk. Target at t is byte t+1 (the same target as
the next-byte head, predicted from strictly less information).

At inference a new boundary yields n speculative bytes (offsets 1..n); they are accepted left to
right while P(byte) >= τ (the first byte of the chunk always comes from the next-byte head).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .relation import SwiGLU


class _MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False, **fk)
        self.out = nn.Linear(d_model, d_model, bias=False, **fk)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B,H,L,dh]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[:, None])  # bool mask [B,1,L,L]
        return self.out(y.transpose(1, 2).reshape(B, L, D))


class _Layer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.norm1 = RMSNorm(d_model, eps=eps, **fk)
        self.attn = _MHA(d_model, n_heads, **fk)
        self.norm2 = RMSNorm(d_model, eps=eps, **fk)
        self.mlp = SwiGLU(d_model, d_ff, **fk)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        return x + self.mlp(self.norm2(x))


def lca_mask(chunk_id: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """chunk_id, valid: [B, L] -> bool [B, L, L]; True where query i may attend key j:
    same chunk with j <= i, or any position of the immediately preceding chunk."""
    ci = chunk_id[:, :, None]
    cj = chunk_id[:, None, :]
    L = chunk_id.shape[1]
    i = torch.arange(L, device=chunk_id.device)
    causal = i[None, :, None] >= i[None, None, :]
    same = (ci == cj) & causal
    prev = cj == (ci - 1)
    allowed = (same | prev) & valid[:, None, :]
    # every query keeps at least itself so softmax never sees an all-masked row
    eye = torch.eye(L, device=chunk_id.device, dtype=torch.bool)[None]
    return allowed | eye


class LCAHead(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, eps: float = 1e-5, max_offset: int = 64, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.max_offset = max_offset
        self.res_proj = nn.Linear(d_model, d_model, bias=False, **fk)  # W_r on the chunk-start encoder state
        self.pos = nn.Embedding(max_offset, d_model, **fk)
        self.layers = nn.ModuleList([_Layer(d_model, n_heads, d_ff, eps, **fk) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model, eps=eps, **fk)

    def build_inputs(self, z: torch.Tensor, chunk_start_state: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """z: [B,L,D] dechunked main output; chunk_start_state: [B,L,D] encoder state at each byte's chunk start;
        offset: [B,L] long position within the chunk."""
        pos = self.pos(offset.clamp(max=self.max_offset - 1))
        return z + self.res_proj(chunk_start_state) + pos.to(z.dtype)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.norm(x)
