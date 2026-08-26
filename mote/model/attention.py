"""Plain causal softmax attention over chunks — the Relation ablation control (2026-08-24).

Parameter-matched to FullRelation: q/k/v/o are the same four d_model x d_model linears as
w1/w2/wi/wo (Relation's extras are ~H/2+1 scalars). Same RoPE tables, same mixer interface,
so `RelationCfg.mixer = "attention"` swaps it into the main network with nothing else changed.
Training and eval use fused SDPA; the cache path exists so checkpoints stay servable.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .relation import _apply_rope, rope_tables

AttentionCache = Tuple[torch.Tensor, torch.Tensor]  # (K [B,H,S,dh], V [B,H,S,dh])


class CausalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layer_idx: int,
        rope_theta: float = 10000.0,
        qk_norm: bool = False,
        device=None,
        dtype=None,
        **_ignored,  # tau_s/lambda_init/givens/window are Relation knobs
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        fk = {"device": device, "dtype": dtype}
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        assert self.d_head % 2 == 0, "head dim must be even for RoPE"
        self.layer_idx = layer_idx
        self.rope_theta = rope_theta
        self.telemetry: Optional[dict] = None
        self.qk_norm = qk_norm
        if qk_norm:  # the control carries it too, so the Relation-vs-attention ablation stays matched
            from .relation import HeadRMSNorm

            self.q_norm = HeadRMSNorm(self.d_head, **fk)
            self.k_norm = HeadRMSNorm(self.d_head, **fk)
        self.wq = nn.Linear(d_model, d_model, bias=False, **fk)
        self.wk = nn.Linear(d_model, d_model, bias=False, **fk)
        self.wv = nn.Linear(d_model, d_model, bias=False, **fk)
        self.wo = nn.Linear(d_model, d_model, bias=False, **fk)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[AttentionCache] = None,
        return_cache: bool = False,
        return_gates: bool = False,
    ):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        S = 0 if cache is None else cache[0].shape[2]

        q = self.wq(x).view(B, T, H, dh).transpose(1, 2)
        k = self.wk(x).view(B, T, H, dh).transpose(1, 2)
        v = self.wv(x).view(B, T, H, dh).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope_tables(S, T, dh, self.rope_theta, x.dtype, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        if S == 0 and not return_gates and self.telemetry is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            u = torch.matmul(q, k.transpose(-1, -2)).float() / math.sqrt(dh)
            qi = torch.arange(S, S + T, device=x.device)
            kj = torch.arange(0, S + T, device=x.device)
            u = u.masked_fill(kj[None, :] > qi[:, None], float("-inf"))
            flow = torch.softmax(u, dim=-1)
            y = torch.matmul(flow.to(v.dtype), v)
        out = self.wo(y.transpose(1, 2).reshape(B, T, D))

        extras = []
        if return_cache:
            extras.append((k, v))
        if return_gates or self.telemetry is not None:
            g = flow.masked_fill(kj[None, :] >= qi[:, None], 0.0).sum(-1)  # mass on strictly-past
            if self.telemetry is not None:
                self.telemetry["exchange_mass"] = float(g[0, :, -1].mean())
            if return_gates:
                extras.append(g)
        return (out, *extras) if extras else out
