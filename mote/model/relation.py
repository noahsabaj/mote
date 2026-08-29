"""Full Relation token mixer — "Ask Self, Ask Others: Relation Is All You Need" (Ge, Yang, Nie 2026).

Per head (width d_h):
    P1 = X W1, P2 = X W2, I = X W_I          (RoPE on P1, P2 only)
    U_ij = p1_i · p2_j / sqrt(d_h)
    S_i  = sigmoid(U_ii / tau_S)                    Self relation   (tau_S = 2)
    E_ij = SiLU(U_ij)                               Exchange relation
    R_ij = S_i (j = i) | E_ij - lambda_l * log(i) (j < i) | -inf (j > i)   (i is 1-based)
    F_i  = softmax(R_i),  Y_i = sum_j F_ij I~_j
Multi-head: information states of adjacent head pairs are mixed by a learnable Givens rotation
before transport; the pairing alternates between layers (M_A = (1,2),(3,4),... ; M_B = (2,3),...,(H,1)).
Decode cache holds only {P2_1:t, I~_1:t} (Appendix A.5).

This is the materialized O(T^2) form. The main network of the H-Net sees ~T/6 positions, so
for 4096-byte contexts T is a few hundred and no fused kernel is needed.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arena import ArenaLayer
from .flash_relation import HAS_TRITON, flash_relation

RelationCache = Tuple[torch.Tensor, torch.Tensor]  # (P2 [B,H,S,dh], I~ [B,H,S,dh]) — or an ArenaLayer (arena.py)
USE_FLASH = True  # fused Triton kernel (flash_relation.py) on CUDA; the materialized path is the reference
FLASH_MIN_T = 16  # below this the launch costs more than the tiny matmuls
_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"  # Triton's CPU interpreter: the kernel path without a GPU (tests)


def _rope_cos_sin(positions: torch.Tensor, dim: int, theta: float, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """positions: [T] int64 -> cos, sin of shape [T, dim] (rotate-half convention, full head)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]  # [T, dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


_ROPE_CACHE: dict = {}


def rope_tables(S: int, T: int, dim: int, theta: float, dtype: torch.dtype, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for absolute positions S..S+T-1, memoised: the same table is needed by every layer
    of every forward, and with chunk-count bucketing only a handful of (S, T) pairs ever occur."""
    key = (S, T, dim, theta, dtype, str(device))
    hit = _ROPE_CACHE.get(key)
    if hit is None:
        if len(_ROPE_CACHE) >= 32:
            _ROPE_CACHE.clear()
        hit = _rope_cos_sin(torch.arange(S, S + T, device=device), dim, theta, dtype)
        _ROPE_CACHE[key] = hit
    return hit


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, dh]; cos/sin: [T, dh]
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


class HeadRMSNorm(nn.Module):
    """RMSNorm over the head dimension, with a learnable per-dimension gain — QK-Norm (Henry et al. 2020).

    A dedicated module rather than `norm.RMSNorm` because that one's fused Triton path is written for the
    residual-stream shape and this runs on [B, H, T, dh]; the arithmetic here is the same reference form,
    in fp32, and the gain is 1-D so it lands in AdamW's no-decay group and is invisible to Muon.

    Why Relation has it at all (2608.24814 §4.2): QK-Norm and learnable RMSNorm gains are the two factors
    that decide how precisely ELR collapse holds — removing QK-Norm takes the collapse error from
    2.3e-3 to 5.2e-3 — and ELR is the coordinate Mote now reads every norm-control gate on. Unlike in
    softmax attention, this is NOT loss-neutral here: Relation's evidence u = p1·p2ᵀ/√dh feeds
    silu(u) − λ·log i and sigmoid(u/τ_s), neither of which is scale-covariant, so the gain has to carry
    the scale back and τ_s / λ were tuned without it. Off by default; gated by an arm.
    """

    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class FullRelation(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layer_idx: int,
        tau_s: float = 2.0,
        lambda_init: float = 0.5,
        rope_theta: float = 10000.0,
        givens: bool = True,
        qk_norm: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % 2 == 0, "Givens pairing needs an even number of heads"
        fk = {"device": device, "dtype": dtype}
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        assert self.d_head % 2 == 0, "head dim must be even for RoPE"
        self.layer_idx = layer_idx
        self.tau_s = tau_s
        self.rope_theta = rope_theta
        self.use_givens = givens
        self.telemetry: Optional[dict] = None  # set to a dict by the serving engine to collect live values

        # QK-Norm on the two evidence projections, before RoPE and before the cache write, so the Triton
        # kernel and the arena both receive already-normalised p2 and need no change of their own.
        self.qk_norm = qk_norm
        if qk_norm:
            self.p1_norm = HeadRMSNorm(self.d_head, **fk)
            self.p2_norm = HeadRMSNorm(self.d_head, **fk)
        self.w1 = nn.Linear(d_model, d_model, bias=False, **fk)
        self.w2 = nn.Linear(d_model, d_model, bias=False, **fk)
        self.wi = nn.Linear(d_model, d_model, bias=False, **fk)
        self.wo = nn.Linear(d_model, d_model, bias=False, **fk)
        # count calibration: one unconstrained fp32 scalar per layer, shared across heads
        self.lam = nn.Parameter(torch.tensor(float(lambda_init), dtype=torch.float32, device=device))
        self.lam._no_weight_decay = True
        if givens:
            # H/2 angles per layer, fp32, zero-init (identity rotation)
            self.theta = nn.Parameter(torch.zeros(n_heads // 2, dtype=torch.float32, device=device))
            self.theta._no_weight_decay = True
            # alternate pairing by layer parity: even -> M_A, odd -> M_B
            if layer_idx % 2 == 0:
                a = torch.arange(0, n_heads, 2)
                b = torch.arange(1, n_heads, 2)
            else:
                a = torch.arange(1, n_heads, 2)
                b = (torch.arange(2, n_heads + 1, 2)) % n_heads
            self.register_buffer("pair_a", a, persistent=False)
            self.register_buffer("pair_b", b, persistent=False)

    # ------------------------------------------------------------------------------
    def _givens(self, info: torch.Tensor) -> torch.Tensor:
        """Rotate information states across adjacent head pairs. info: [B, H, T, dh]."""
        if not self.use_givens:
            return info
        B, H, T, dh = info.shape
        c = torch.cos(self.theta).to(info.dtype)[None, :, None, None]
        s = torch.sin(self.theta).to(info.dtype)[None, :, None, None]
        # even layers pair (0,1),(2,3),...; odd layers pair (1,2),...,(H-1,0): roll the head axis so
        # both are the same adjacent-pair view. Same arithmetic as the index/scatter form, no clone.
        odd = self.layer_idx % 2 == 1
        x = torch.roll(info, -1, dims=1) if odd else info
        x = x.view(B, H // 2, 2, T, dh)
        ia, ib = x[:, :, 0], x[:, :, 1]
        out = torch.stack([c * ia - s * ib, s * ia + c * ib], dim=2).view(B, H, T, dh)
        return torch.roll(out, 1, dims=1) if odd else out

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[RelationCache] = None,
        return_cache: bool = False,
        return_gates: bool = False,
    ):
        """x: [B, T, D]. With a cache, x holds the new T positions appended after S cached ones.

        Returns out [B, T, D] and optionally the updated cache and the per-position exchange
        mass g_i = sum_{j<i} F_ij (telemetry: how much each token draws from history).
        """
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        arena = isinstance(cache, ArenaLayer)  # append-only rows in the shared decode arena (arena.py)
        if arena:
            S = cache.n
        else:
            S = 0 if cache is None else cache[0].shape[2]

        p1 = self.w1(x).view(B, T, H, dh).transpose(1, 2)
        p2 = self.w2(x).view(B, T, H, dh).transpose(1, 2)
        info = self.wi(x).view(B, T, H, dh).transpose(1, 2)

        if self.qk_norm:
            p1 = self.p1_norm(p1)
            p2 = self.p2_norm(p2)
        cos, sin = rope_tables(S, T, dh, self.rope_theta, x.dtype, x.device)
        p1 = _apply_rope(p1, cos, sin)
        p2 = _apply_rope(p2, cos, sin)
        info = self._givens(info)

        if arena:
            cache.write(p2, info)  # rows [S, S+T); the caller advances n after every layer wrote
            p2_all, info_all = cache.views(S + T)
        elif cache is not None:
            p2_all = torch.cat([cache[0], p2], dim=2)
            info_all = torch.cat([cache[1], info], dim=2)
        else:
            p2_all, info_all = p2, info
        Sall = S + T
        new_cache = cache if arena else (p2_all, info_all)

        if USE_FLASH and HAS_TRITON and (x.is_cuda or _INTERPRET) and T >= FLASH_MIN_T:
            y, g = flash_relation(p1, p2_all, info_all, self.lam, self.tau_s, q_start=S)
            out = self.wo(y.transpose(1, 2).reshape(B, T, D))
            extras = []
            if return_cache:
                extras.append(new_cache)
            if self.telemetry is not None:
                self.telemetry["exchange_mass"] = float(g[0, :, -1].detach().mean())  # also runs under grad (RL updates)
            if return_gates:
                extras.append(g)
            return (out, *extras) if extras else out

        # pairwise evidence in fp32 (materialized reference path)
        u = torch.matmul(p1, p2_all.transpose(-1, -2)).float() / math.sqrt(dh)  # [B,H,T,Sall]
        qi = torch.arange(S, Sall, device=x.device)  # absolute query index (0-based)
        kj = torch.arange(0, Sall, device=x.device)
        is_self = qi[:, None] == kj[None, :]
        is_past = kj[None, :] < qi[:, None]
        log_i = torch.log((qi + 1).float())[:, None]  # 1-based count correction
        r = torch.where(
            is_self,
            torch.sigmoid(u / self.tau_s),
            F.silu(u) - self.lam.float() * log_i,
        )
        r = r.masked_fill(~(is_self | is_past), float("-inf"))
        flow = torch.softmax(r, dim=-1)
        y = torch.matmul(flow.to(info_all.dtype), info_all)  # [B,H,T,dh]
        out = self.wo(y.transpose(1, 2).reshape(B, T, D))

        extras = []
        if return_cache:
            extras.append(new_cache)
        if return_gates or self.telemetry is not None:
            g = flow.masked_fill(~is_past, 0.0).sum(-1)  # [B,H,T] exchange mass
            if self.telemetry is not None:
                self.telemetry["exchange_mass"] = float(g[0, :, -1].detach().mean())  # also runs under grad (RL updates)
            if return_gates:
                extras.append(g)
        if extras:
            return (out, *extras)
        return out


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.fc1 = nn.Linear(d_model, 2 * d_ff, bias=False, **fk)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False, **fk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, g = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(g) * a)
