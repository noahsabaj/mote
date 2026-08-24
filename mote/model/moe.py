"""Mixture-of-experts SwiGLU for the main network (signed 2026-08-24, docs/shape.md "MoE").

A drop-in for `SwiGLU` in the Relation blocks: E experts of hidden width f (half the dense width in the
signed layouts, so top-2 routing costs the dense FFN's FLOPs), all expert weights in ONE stacked tensor per
matrix (`w1` [E, d, 2f], `w2` [E, f, d]) so Muon orthogonalises them in one batched Newton-Schulz launch.
Parameters without activations: top-k routing adds no activation memory, which is the property that makes
MoE the one way to add parameters on an 8 GB card at 16384 context (docs/research/moe-2026-08-24.md).

Three execution paths, exact-equal up to summation order (tests/test_moe.py):
  dense    every expert on every token, masked by the gate matrix — static shapes, CUDA-graph capturable.
           Decode (one chunk per byte) and tiny inputs take it; also the CPU reference.
  loop     per-expert gather / matmul / index_add — the CPU training path.
  grouped  tokens sorted by expert, two `torch.nn.functional.grouped_mm` launches per layer with jagged
           rows per expert (bf16, CUDA, SM ≥ 80) — the GPU training path.

Routers (`RelationCfg.moe_router`):
  lossfree DeepSeek-V3 (2412.19437) / Moonlight (2502.16982): sigmoid affinities s, selection on s + b where
           b is a per-expert bias BUFFER moved after every optimizer step towards balanced load (never by
           gradient), gate weights = the selected s renormalised over top-k times the gate scale
           (Moonlight Fig. 6: the factor that gives a dense FFN's output RMS; DeepSeek-V3 used 2.5,
           Moonlight 2.446 at 64/6 — computed here for our E/k), plus the sequence-level balance loss with
           a small weight (the Kakao 2608.20061 reference recipe keeps it).
  aux      Switch / GShard: softmax affinities, selected renormalised, load-balance loss α·E·Σ_i f_i·P_i and
           a router z-loss — the setup the joint MoE scaling law (2502.05172) was fit on.
The router runs in fp32 outside autocast; padded chunk rows (`token_mask` False) are excluded from every
load statistic and loss.

Telemetry per layer per step, device tensors (no sync): `stats` = load per expert (sums to 1), MaxVio
(2608.20061: (max load − mean) / mean), top-k probability mass; `last_expert_out` keeps the per-expert
outputs when `keep_expert_out` (2608.17687 measured 2.5× decode cost to retrofit this). Router entropy is
deliberately not a confidence signal (2608.17687: sign flips between models).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_GROUPED = hasattr(F, "grouped_mm")


def gate_scale(n_experts: int, top_k: int, draws: int = 20000, seed: int = 0) -> float:
    """Moonlight's gate scaling factor (2502.16982, Fig. 6): E[1 / sqrt(Σ p_i²)] over Gaussian router logits,
    p = the top-k sigmoid affinities renormalised — the multiplier that gives the routed sum the RMS of a
    single dense FFN. 2.446 at E=64/k=6 (theirs); ≈1.3 at E=4/k=2, ≈1.35 at E=8/k=2 (ours)."""
    if top_k >= n_experts and top_k == 1:
        return 1.0
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(draws, n_experts, generator=g)
    p = torch.sigmoid(logits).topk(min(top_k, n_experts), dim=-1).values
    p = p / p.sum(-1, keepdim=True)
    return float((1.0 / p.pow(2).sum(-1).sqrt()).mean())


class MoESwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int = 2, router: str = "lossfree",
                 aux_weight: Optional[float] = None, z_weight: float = 1e-3, bias_gamma: float = 1e-3,
                 scale: Optional[float] = None, dense_threshold: int = 16, device=None, dtype=None):
        super().__init__()
        assert n_experts >= 2 and 1 <= top_k <= n_experts, (n_experts, top_k)
        assert router in ("lossfree", "aux"), router
        fk = {"device": device, "dtype": dtype}
        self.d_model, self.d_ff, self.n_experts, self.top_k = d_model, d_ff, n_experts, top_k
        self.router_kind = router
        self.aux_weight = (1e-4 if router == "lossfree" else 1e-2) if aux_weight is None else float(aux_weight)
        self.z_weight = float(z_weight)
        self.bias_gamma = float(bias_gamma)
        self.scale = (gate_scale(n_experts, top_k) if router == "lossfree" else 1.0) if scale is None else float(scale)
        self.dense_threshold = dense_threshold
        self.keep_expert_out = False
        self.last_expert_out: Optional[torch.Tensor] = None
        # expert stacks: nn.Linear's default init per expert (U(±1/sqrt(fan_in))) so an E=1 stack equals SwiGLU
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, 2 * d_ff, **fk))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model, **fk))
        with torch.no_grad():
            self.w1.uniform_(-1.0 / math.sqrt(d_model), 1.0 / math.sqrt(d_model))
            self.w2.uniform_(-1.0 / math.sqrt(d_ff), 1.0 / math.sqrt(d_ff))
        self.w1._muon_stack = True  # Muon orthogonalises each [d, 2f] slice; the stack is one batched launch
        self.w2._muon_stack = True
        self.router = nn.Linear(d_model, n_experts, bias=False, device=device, dtype=torch.float32)
        self.router.weight._no_muon = True  # AdamW, like Mamba-3's in_proj: not a hidden square-ish map
        self.register_buffer("expert_bias", torch.zeros(n_experts, device=device, dtype=torch.float32))
        self.register_buffer("_load_acc", torch.zeros(n_experts, device=device, dtype=torch.float32), persistent=False)
        self.stats: Dict[str, torch.Tensor] = {}
        self.aux_loss: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------------------
    def active_expert_params(self) -> int:
        """Parameters a token actually touches: k of the E experts plus the router."""
        per = self.w1[0].numel() + self.w2[0].numel()
        return self.top_k * per + self.router.weight.numel()

    def extra_repr(self) -> str:
        return f"experts={self.n_experts}, top_k={self.top_k}, d_ff={self.d_ff}, router={self.router_kind}, scale={self.scale:.3f}"

    # ------------------------------------------------------------------------------
    def _route(self, xf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """fp32 routing: (logits [N,E], normalised affinities [N,E], top-k indices [N,k], gate weights [N,k])."""
        with torch.autocast(device_type=xf.device.type, enabled=False):
            logits = F.linear(xf.float(), self.router.weight)
            if self.router_kind == "lossfree":
                s = torch.sigmoid(logits)
                topi = (s + self.expert_bias).topk(self.top_k, dim=-1).indices  # the bias picks, never weighs
                w = s.gather(1, topi)
                w = w / w.sum(-1, keepdim=True).clamp_min(1e-9) * self.scale
                p_full = s / s.sum(-1, keepdim=True).clamp_min(1e-9)
            else:
                p_full = torch.softmax(logits, dim=-1)
                topi = p_full.topk(self.top_k, dim=-1).indices
                w = p_full.gather(1, topi)
                w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        return logits, p_full, topi, w

    def _balance(self, logits: torch.Tensor, p_full: torch.Tensor, topi: torch.Tensor, valid: torch.Tensor, rows: int) -> None:
        """Load statistics and the balance loss over the valid tokens; `rows` = sequences in the batch so
        the loss is sequence-level (DeepSeek-V3 eq. 17–20) when the input was [B, M, D]."""
        E, k = self.n_experts, self.top_k
        onehot = torch.zeros(topi.shape[0], E, device=topi.device, dtype=torch.float32).scatter_(1, topi, 1.0)
        onehot = onehot * valid[:, None]
        n_valid = valid.sum().clamp_min(1.0)
        counts = onehot.sum(0)  # tokens per expert (every valid token counts k times in total)
        load = counts / (n_valid * k)
        mean = load.mean()
        maxvio = (load.max() - mean) / mean.clamp_min(1e-9)
        # per-sequence f_i and P_i, averaged over sequences (batch-level when rows == 1)
        oh = onehot.view(rows, -1, E)
        pv = (p_full * valid[:, None]).view(rows, -1, E)
        nv = valid.view(rows, -1).sum(1).clamp_min(1.0)  # [B]
        f = oh.sum(1) * E / (nv[:, None] * k)  # Switch's f_i (mean 1 when balanced)
        P = pv.sum(1) / nv[:, None]
        bal = (f * P).sum(1).mean()
        if self.router_kind == "lossfree":
            aux = self.aux_weight * bal
        else:
            z = (torch.logsumexp(logits, dim=-1).pow(2) * valid).sum() / n_valid
            aux = self.aux_weight * bal + self.z_weight * z
        soft = torch.softmax(logits, dim=-1).gather(1, topi).sum(-1)  # comparable across routers
        self.aux_loss = aux
        self.stats = {"load": load.detach(), "maxvio": maxvio.detach(), "topk_mass": ((soft * valid).sum() / n_valid).detach()}
        if self.training and self.router_kind == "lossfree":
            self._load_acc += counts.detach()

    @torch.no_grad()
    def update_bias(self) -> None:
        """DeepSeek-V3's bias step after an optimizer step: overloaded experts go down by γ, underloaded up,
        centred as in Moonlight (mean sign subtracted) so the bias vector does not drift as a whole."""
        if self.router_kind != "lossfree" or self.bias_gamma <= 0:
            return
        load = self._load_acc
        if load.sum() <= 0:
            return
        e = (load.mean() - load) / load.mean().clamp_min(1e-9)  # > 0 when underloaded
        sgn = torch.sign(e)
        self.expert_bias += self.bias_gamma * (sgn - sgn.mean())
        self._load_acc.zero_()

    # ------------------------------------------------------------------------------
    def _dense(self, xf: torch.Tensor, topi: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("nd,edf->nef", xf, self.w1)
        a, g = h.chunk(2, dim=-1)
        y = torch.einsum("nef,efd->ned", F.silu(g) * a, self.w2)  # [N, E, d]
        if self.keep_expert_out:
            self.last_expert_out = y.detach()
        gate = torch.zeros(xf.shape[0], self.n_experts, device=y.device, dtype=y.dtype).scatter_(1, topi, w.to(y.dtype))
        return (y * gate[..., None]).sum(1)

    def _loop(self, xf: torch.Tensor, topi: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        out = None
        for e in range(self.n_experts):
            m = topi == e
            rows = m.any(1).nonzero(as_tuple=False).squeeze(1)
            if rows.numel() == 0:
                continue
            we = (w * m).sum(1)[rows]
            h = xf[rows] @ self.w1[e]
            a, g = h.chunk(2, dim=-1)
            y = (F.silu(g) * a) @ self.w2[e]
            if out is None:
                out = torch.zeros(xf.shape[0], self.d_model, device=y.device, dtype=y.dtype)
            out = out.index_add(0, rows, y * we.to(y.dtype)[:, None])
        if out is None:
            out = torch.zeros(xf.shape[0], self.d_model, device=xf.device, dtype=xf.dtype)
        return out

    def _grouped(self, xf: torch.Tensor, topi: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        N, k, E = xf.shape[0], self.top_k, self.n_experts
        flat = topi.reshape(-1)
        order = torch.argsort(flat, stable=True)
        tok = order // k
        offs = torch.bincount(flat, minlength=E).cumsum(0).to(torch.int32)
        xs = xf.index_select(0, tok).to(torch.bfloat16)
        h = F.grouped_mm(xs, self.w1.to(torch.bfloat16), offs=offs)
        a, g = h.chunk(2, dim=-1)
        y = F.grouped_mm((F.silu(g) * a).to(torch.bfloat16), self.w2.to(torch.bfloat16), offs=offs)
        y = y * w.reshape(-1)[order].to(y.dtype)[:, None]
        return torch.zeros(N, self.d_model, device=y.device, dtype=y.dtype).index_add_(0, tok, y)

    def forward(self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        N = xf.shape[0]
        rows = shape[0] if x.dim() == 3 else 1
        logits, p_full, topi, w = self._route(xf)
        valid = token_mask.reshape(-1).to(torch.float32) if token_mask is not None else torch.ones(N, device=xf.device)
        self._balance(logits, p_full, topi, valid, rows)
        if N <= self.dense_threshold:
            y = self._dense(xf, topi, w)
        elif _HAS_GROUPED and xf.is_cuda and torch.is_autocast_enabled():
            y = self._grouped(xf, topi, w)
        else:
            y = self._loop(xf, topi, w)
        return y.reshape(*shape[:-1], self.d_model)


def moe_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, MoESwiGLU)]


def collect_moe(model: nn.Module) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """(summed balance/z losses of the last forward, telemetry) — every value a 0-dim device tensor."""
    mods = moe_modules(model)
    if not mods:
        return None, {}
    aux = sum(m.aux_loss for m in mods if m.aux_loss is not None)
    mv = torch.stack([m.stats["maxvio"] for m in mods])
    stats = {"moe_aux": aux.detach(), "moe_maxvio": mv.mean(), "moe_maxvio_max": mv.max(),
             "moe_topk_mass": torch.stack([m.stats["topk_mass"] for m in mods]).mean()}
    return aux, stats
