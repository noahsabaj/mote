"""Multi-byte prediction head with Latent Causal Attention (Owodunni et al. 2026), adapted to H-Net.

For the byte at position t inside chunk k (offset m = t - start_k) the head's input is
    x_t = z_k + W_r x̂_{start_k} + pos(m)
where z_k is the main network's (dechunked) output for chunk k and x̂_{start_k} the encoder state
at the chunk's first byte — everything depends only on bytes <= start_k, so all bytes of a chunk
can be predicted in parallel. The LCA mask lets position t attend to earlier positions of its own
chunk and to every position of the previous chunk. Target at t is byte t+1 (the same target as
the next-byte head, predicted from strictly less information).

The LCA mask is exactly the contiguous causal window  lo(t) ≤ j ≤ t  with lo(t) = start of the
previous chunk, a few bytes wide. `block_local_attention` exploits that: queries in a block of W
bytes only see the key blocks covering their windows, so the score work is O(L·W) instead of O(L²)
and the result equals the dense masked attention exactly (same allowed set, same softmax).

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

BLOCK = 64  # byte block for the local attention
USE_BLOCK_LOCAL = True  # dense masked SDPA remains the reference (tests, tiny sequences)


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


def window_lo(chunk_id: torch.Tensor) -> torch.Tensor:
    """Lowest key position each query may see: the start of the previous chunk (0 for chunk 0). [B, L]."""
    B, L = chunk_id.shape
    pos = torch.arange(L, device=chunk_id.device)[None, :].expand(B, -1)
    is_start = torch.ones_like(chunk_id, dtype=torch.bool)
    is_start[:, 1:] = chunk_id[:, 1:] != chunk_id[:, :-1]
    n_chunks = int(chunk_id.max()) + 1
    starts = torch.zeros(B, n_chunks, dtype=torch.long, device=chunk_id.device)
    starts = starts.scatter_reduce(1, chunk_id, torch.where(is_start, pos, torch.zeros_like(pos)), reduce="amax", include_self=True)
    return starts.gather(1, (chunk_id - 1).clamp(min=0))


def block_local_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_id: torch.Tensor, valid: torch.Tensor, W: int = BLOCK) -> torch.Tensor:
    """q, k, v: [B, H, L, dh]. Exact LCA attention via blocks: query block b attends to key blocks
    b-nb+1 .. b, where nb is chosen so every query's window fits. Returns [B, H, L, dh]."""
    B, H, L, dh = q.shape
    lo = window_lo(chunk_id)  # [B, L]
    pos = torch.arange(L, device=q.device)[None, :].expand(B, -1)
    span = int((pos - lo).max())
    nb = span // W + 2  # key blocks per query block: enough for the widest window, +1 for alignment
    P = (-L) % W
    if P:
        pad = lambda t: F.pad(t, (0, 0, 0, P))
        q, k, v = pad(q), pad(k), pad(v)
        valid = F.pad(valid, (0, P), value=False)
        lo = F.pad(lo, (0, P), value=0)
        pos = F.pad(pos, (0, P), value=L + P)  # padded queries see nothing but themselves
    Lp = L + P
    nq = Lp // W
    qb = q.view(B, H, nq, W, dh)
    kb = k.view(B, H, nq, W, dh)
    vb = v.view(B, H, nq, W, dh)
    # gather key blocks b-nb+1..b for each query block b (zero blocks before the sequence start)
    kp = F.pad(kb, (0, 0, 0, 0, nb - 1, 0))  # [B,H,nq+nb-1,W,dh]
    vp = F.pad(vb, (0, 0, 0, 0, nb - 1, 0))
    kw = kp.unfold(2, nb, 1)  # [B,H,nq,W,dh,nb]
    vw = vp.unfold(2, nb, 1)
    kw = kw.permute(0, 1, 2, 5, 3, 4).reshape(B, H, nq, nb * W, dh)
    vw = vw.permute(0, 1, 2, 5, 3, 4).reshape(B, H, nq, nb * W, dh)
    # absolute key positions for each query block: (b-nb+1)*W + j
    b_idx = torch.arange(nq, device=q.device)
    j_abs = (b_idx[:, None] - (nb - 1)) * W + torch.arange(nb * W, device=q.device)[None, :]  # [nq, nb*W]
    i_abs = pos.view(B, nq, W)  # [B,nq,W]
    lo_b = lo.view(B, nq, W)
    j_in = (j_abs >= 0) & (j_abs < Lp)
    j_safe = j_abs.clamp(min=0, max=Lp - 1)
    valid_j = valid.gather(1, j_safe.reshape(1, -1).expand(B, -1)).view(B, nq, nb * W) & j_in[None]
    ja = j_abs[None, :, None, :]  # [1,nq,1,nb*W]
    ia = i_abs[:, :, :, None]  # [B,nq,W,1]
    allowed = (ja <= ia) & (ja >= lo_b[:, :, :, None]) & valid_j[:, :, None, :]
    allowed = allowed | (ja == ia)  # every query keeps itself
    y = F.scaled_dot_product_attention(qb, kw, vw, attn_mask=allowed[:, None])  # [B,H,nq,W,dh]
    return y.reshape(B, H, Lp, dh)[:, :, :L]


class _MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False, **fk)
        self.out = nn.Linear(d_model, d_model, bias=False, **fk)

    def forward(self, x: torch.Tensor, chunk_id: torch.Tensor, valid: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.d_head).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B,H,L,dh]
        if USE_BLOCK_LOCAL and L >= 2 * BLOCK:
            y = block_local_attention(q, k, v, chunk_id, valid)
        else:
            if attn_mask is None:
                attn_mask = lca_mask(chunk_id, valid)
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

    def forward(self, x: torch.Tensor, chunk_id: torch.Tensor, valid: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), chunk_id, valid, attn_mask)
        return x + self.mlp(self.norm2(x))


class LCAHead(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, eps: float = 1e-5, max_offset: int = 64, vocab: int = 0, transition: bool = False, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.max_offset = max_offset
        # DSpark-style sequential correction: a bias on the logits for byte t+1 given byte t. At the byte
        # vocabulary the exact V×V table is 70K parameters, so no low-rank factorisation is needed. In
        # training it is teacher-forced (the real previous byte); at decode it conditions each draft slot
        # on the draft byte actually sampled before it, which parallel drafting otherwise lacks.
        self.transition = nn.Embedding(vocab, vocab, device=device, dtype=torch.float32) if (transition and vocab) else None
        if self.transition is not None:
            nn.init.zeros_(self.transition.weight)
            self.transition.weight._no_reinit = True
            self.transition.weight._no_weight_decay = True
        self.res_proj = nn.Linear(d_model, d_model, bias=False, **fk)  # W_r on the chunk-start encoder state
        self.pos = nn.Embedding(max_offset, d_model, **fk)
        self.layers = nn.ModuleList([_Layer(d_model, n_heads, d_ff, eps, **fk) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model, eps=eps, **fk)

    def build_inputs(self, z: torch.Tensor, chunk_start_state: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """z: [B,L,D] dechunked main output; chunk_start_state: [B,L,D] encoder state at each byte's chunk start;
        offset: [B,L] long position within the chunk."""
        pos = self.pos(offset.clamp(max=self.max_offset - 1))
        return z + self.res_proj(chunk_start_state) + pos.to(z.dtype)

    def forward(self, x: torch.Tensor, chunk_id: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: [B,L,D]; chunk_id, valid: [B,L]. The dense mask is built once only when the dense path runs."""
        B, L, _ = x.shape
        attn_mask = None if (USE_BLOCK_LOCAL and L >= 2 * BLOCK) else lca_mask(chunk_id, valid)
        for layer in self.layers:
            x = layer(x, chunk_id, valid, attn_mask)
        return self.norm(x)
