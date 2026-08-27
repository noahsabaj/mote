"""Dynamic chunking — routing, chunk, dechunk, straight-through residual, ratio loss.

Follows the official H-Net implementation (goombalab/hnet, hnet/modules/dc.py) with two
additions: a pure-PyTorch EMA path when the Mamba-2 chunk-scan kernel is unavailable, and the
ATDC target-ratio schedule (Dang et al. 2026).

Shapes use the padded batch mode (B, L, D) with a boolean `mask`; packed/cu_seqlens mode is
not needed for our training (fixed windows) or inference (batch 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Linux / WSL2 with Triton: reuse the Mamba-2 chunk scan for the EMA (as upstream does)
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # type: ignore

    HAS_SSD_KERNEL = True
except Exception:  # pragma: no cover - Windows-native fallback
    mamba_chunk_scan_combined = None
    HAS_SSD_KERNEL = False


# --------------------------------------------------------------------------------------
@dataclass
class RoutingOutput:
    boundary_prob: torch.Tensor  # [B, L, 2]  (p(no boundary), p(boundary))
    boundary_mask: torch.Tensor  # [B, L] bool
    selected_probs: torch.Tensor  # [B, L, 1]  confidence c_t of the taken decision


@dataclass
class RoutingState:
    has_seen_tokens: torch.Tensor  # [B] bool
    last_hidden_state: torch.Tensor  # [B, D]


@dataclass
class DeChunkState:
    last_value: torch.Tensor  # [B, D]


class RoutingModule(nn.Module):
    """p_t = 1/2 (1 - cos(q_t, k_{t-1})) with q/k projections initialized to identity."""

    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **fk)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **fk)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True
        # Bounded routing + the decode threshold. `target_ratio` is the live N (the trainer sets it
        # every step from the ATDC schedule); `bound_*` come from the config and default to off, so a
        # model that declares nothing routes exactly as it did before 2026-08-27.
        self.target_ratio: float = 5.0
        self.bound_floor: int = 0
        self.bound_ceiling: Optional[float] = None
        self.decode_threshold: float = 0.5
        # Below this many positions a "budget" is not a meaningful quantity — a 3-byte speculative
        # round would be forced to produce one boundary. Prefill windows are thousands of bytes; the
        # short continuations (drafts, tool-result injection) fall under it and route unbounded.
        self.bound_min_len: int = 64

    def allocate_inference_cache(self, batch_size: int, device, dtype=None) -> RoutingState:
        return RoutingState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor, state: Optional[RoutingState] = None) -> RoutingOutput:
        # hidden: [B, L, D]; mask: [B, L] bool (valid positions)
        continuing = state is not None and bool(state.has_seen_tokens.any())
        if continuing:
            # continue a sequence: the first position is compared with the last byte seen before it
            assert bool(state.has_seen_tokens.all()), "mixed fresh/continuing rows are not supported"
            prev = torch.cat([state.last_hidden_state[:, None].to(hidden.dtype), hidden[:, :-1]], dim=1)
            q = F.normalize(self.q_proj_layer(prev), dim=-1)
            k = F.normalize(self.k_proj_layer(hidden), dim=-1)
            p = torch.clamp((1 - (q * k).sum(-1)) / 2, min=0.0, max=1.0)  # [B, L]
        else:
            q = F.normalize(self.q_proj_layer(hidden[:, :-1]), dim=-1)
            k = F.normalize(self.k_proj_layer(hidden[:, 1:]), dim=-1)
            cos_sim = (q * k).sum(-1)  # [B, L-1]
            p = torch.clamp((1 - cos_sim) / 2, min=0.0, max=1.0)
            p = F.pad(p, (1, 0), value=1.0)  # first position is always a boundary
        boundary_prob = torch.stack([1 - p, p], dim=-1)
        selected_idx = boundary_prob.argmax(dim=-1)
        boundary_mask = (selected_idx == 1) & mask
        boundary_mask = self._bound(boundary_mask, p, mask)
        selected_idx = boundary_mask.long()
        if state is not None:
            has = mask.any(dim=-1)
            state.has_seen_tokens.copy_(has | state.has_seen_tokens)
            last = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            state.last_hidden_state.copy_(
                torch.where(has[:, None], hidden[torch.arange(hidden.shape[0], device=hidden.device), last], state.last_hidden_state)
            )
        selected_probs = boundary_prob.gather(-1, selected_idx.unsqueeze(-1))
        return RoutingOutput(boundary_prob, boundary_mask, selected_probs)

    def _bound(self, boundary_mask: torch.Tensor, p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Project the mask onto this router's token budget, when one is configured and the window is
        long enough for it to mean anything."""
        if self.bound_ceiling is None and self.bound_floor <= 0:
            return boundary_mask
        if boundary_mask.shape[-1] < self.bound_min_len:
            return boundary_mask
        valid = mask.sum(dim=-1)
        target = valid.float() / max(float(self.target_ratio), 1e-6)  # K = tau * L, tau = 1/N
        k_max = ((target * self.bound_ceiling).ceil().long() if self.bound_ceiling is not None
                 else torch.full_like(valid, boundary_mask.shape[-1]))
        k_min = torch.full_like(valid, int(self.bound_floor)).clamp(max=boundary_mask.shape[-1])
        return project_boundaries(boundary_mask, p, mask, k_min, torch.maximum(k_max, k_min))

    def step(self, hidden: torch.Tensor, state: RoutingState) -> RoutingOutput:
        # hidden: [B, 1, D]
        h = hidden.squeeze(1)
        q = F.normalize(self.q_proj_layer(state.last_hidden_state), dim=-1)
        k = F.normalize(self.k_proj_layer(h), dim=-1)
        p = torch.clamp((1 - (q * k).sum(-1)) / 2, min=0.0, max=1.0)
        state.last_hidden_state.copy_(h)
        p = torch.where(state.has_seen_tokens, p, torch.ones_like(p))
        state.has_seen_tokens.fill_(True)
        boundary_prob = torch.stack([1 - p, p], dim=-1)  # [B, 2]
        # One byte, no sequence to project over: decode keeps a plain threshold. 0.5 is H-Net's
        # (cos_sim < 0), and a bounded-routing run calibrates this to the rate its projection
        # actually produced — see config.py::DCCfg.decode_threshold.
        sel = boundary_prob[..., 1] > self.decode_threshold
        return RoutingOutput(boundary_prob, sel, boundary_prob.max(dim=-1).values.unsqueeze(-1))


def project_boundaries(boundary_mask: torch.Tensor, p: torch.Tensor, mask: torch.Tensor,
                       k_min: torch.Tensor, k_max: torch.Tensor) -> torch.Tensor:
    """Bounded routing (GeneZip 2602.17739 §3.3): flip the fewest boundary decisions, by confidence,
    to land each row's boundary count inside [k_min, k_max].

    The paper states it as two cases — "if too few boundaries are selected, activate the largest-p
    inactive positions; if too many, deactivate the smallest-p active positions" — and both collapse
    into ONE ranking. Score every valid position by p, with a constant offset that puts every active
    position above every inactive one, then keep the top k where k is the original count clamped to
    the interval:

        too many   k = k_max   keeps the k_max highest-confidence actives, drops the rest
        too few    k = k_min   keeps every active (they all outrank inactives) and promotes the
                               highest-confidence inactives to make up the deficit
        in range   k = n       the ranking is a no-op and the router's own decision stands

    Mote's first position always carries p = 1.0 (dc.py pads it), so it can never be the one dropped,
    which is the paper's "the first position in each segment is always kept" for free.

    No gradient flows here and none is lost: `boundary_mask` is a bool tensor with no grad path
    already (see ratio_loss), and the differentiable half of chunking runs through `boundary_prob`.
    """
    b = boundary_mask & mask
    n = b.sum(dim=-1)
    k = torch.minimum(torch.maximum(n, k_min), k_max).clamp(min=0)
    if bool((k == n).all()):
        return b
    score = torch.where(b, p + 2.0, p)  # every active outranks every inactive
    score = torch.where(mask, score, torch.full_like(score, -1e9))
    order = score.argsort(dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(order.shape[-1], device=order.device).expand_as(order))
    return (rank < k.unsqueeze(-1)) & mask


class ChunkLayer(nn.Module):
    """Keep only boundary positions; compact them to the front of a [B, M] tensor with a mask.

    `bucket` rounds M up to a multiple (capped at L). The padded tail holds non-boundary bytes in
    order; the main network is causal, so nothing valid can see them, and dechunk never reads them.
    Stable shapes are what let Triton autotune caches, CUDA graphs and the allocator work.
    """

    def __init__(self, bucket: int = 1):
        super().__init__()
        self.bucket = max(int(bucket), 1)

    def forward(self, hidden: torch.Tensor, boundary_mask: torch.Tensor, exact: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = hidden.shape
        num = boundary_mask.sum(dim=-1)
        M = int(num.max())
        if self.bucket > 1 and not exact:
            M = min(L, -(-M // self.bucket) * self.bucket)
        token_idx = torch.arange(L, device=hidden.device)[None, :] + (~boundary_mask).long() * L
        order = torch.argsort(token_idx, dim=1)[:, :M]
        out = torch.gather(hidden, 1, order[:, :, None].expand(-1, -1, D))
        next_mask = torch.arange(M, device=hidden.device)[None, :] < num[:, None]
        return out, next_mask

    def step(self, hidden: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
        # hidden [B, 1, D] -> [B', 1, D] rows where a boundary fired
        return hidden[boundary_mask]


class DeChunkLayer(nn.Module):
    """EMA smoothing z̄_t = p_t ẑ_t + (1 - p_t) z̄_{t-1} over chunk outputs, then expand to bytes."""

    def __init__(self, d_model: int, headdim: int = 32, block_size: int = 256, prob_clamp: float = 1e-4):
        super().__init__()
        self.d_model = d_model
        self.headdim = headdim
        self.block_size = block_size
        self.eps = prob_clamp
        assert d_model % headdim == 0
        self.nheads = d_model // headdim
        self.use_kernel = False  # the Triton SSD kernel re-autotunes for every new chunk count; chunked torch is faster for us

    def allocate_inference_cache(self, batch_size: int, device, dtype=None) -> DeChunkState:
        return DeChunkState(last_value=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype))

    @staticmethod
    def _ema_chunked(x: torch.Tensor, p: torch.Tensor, C: int = 64, init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Stable parallel EMA: z̄_t = p_t x_t + (1-p_t) z̄_{t-1}, evaluated per block of C steps.

        Within a block, z̄_t = Σ_{s≤t} exp(L_t - L_s) p_s x_s + exp(L_t - L_0) z̄_0 with L_t = Σ_{u≤t} log(1-p_u);
        all exponents are ≤ 0 so nothing overflows. O(M·C·D) work, no sequential Python loop over M.
        """
        B, M, D = x.shape
        wdt = torch.float64 if x.dtype == torch.float64 else torch.float32
        xf, pf = x.to(wdt), p.to(wdt)
        pad = (-M) % C
        if pad:
            xf = F.pad(xf, (0, 0, 0, pad))
            pf = F.pad(pf, (0, pad))  # p=0 on padding: state is carried unchanged
        nb = xf.shape[1] // C
        xf = xf.view(B, nb, C, D)
        pf = pf.view(B, nb, C)
        logq = torch.log1p(-pf)  # log(1-p) ≤ 0
        L = torch.cumsum(logq, dim=-1)  # [B, nb, C]
        # within-block transfer matrix A[t, s] = exp(L_t - L_s) for s <= t
        diff = L[..., :, None] - L[..., None, :]  # [B, nb, C, C]
        mask = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.bool))
        A = torch.exp(diff.masked_fill(~mask, float("-inf")))
        local = torch.matmul(A, pf[..., None] * xf)  # [B, nb, C, D]
        decay_t = torch.exp(L)  # exp(L_t - L_0) with L_0 = 0 at block start  [B, nb, C]
        # Carry across blocks in closed form (no Python loop over blocks: static shapes, one launch):
        # the value entering block b is  Σ_{c<b} exp(S_b − S_c⁺) e_c + exp(S_b) init,  with e_c the
        # block-c end value from its own terms, S_b = Σ_{j<b} log d_j the log-decay before block b and
        # S_c⁺ = Σ_{j≤c} log d_j. All exponents are ≤ 0, as within a block.
        e = local[:, :, -1]  # [B, nb, D]
        logd = L[:, :, -1]  # [B, nb]  log of each block's total decay
        s_incl = torch.cumsum(logd, dim=-1)  # S_c⁺
        s_excl = s_incl - logd  # S_b
        w = s_excl[:, :, None] - s_incl[:, None, :]  # [B, nb, nb]: rows b, cols c
        lower = torch.tril(torch.ones(nb, nb, device=x.device, dtype=torch.bool), diagonal=-1)  # c < b
        w = torch.exp(w.masked_fill(~lower, float("-inf")))
        carry_in = torch.matmul(w, e)  # [B, nb, D]
        if init is not None:
            carry_in = carry_in + torch.exp(s_excl)[..., None] * init.to(wdt)[:, None, :]
        out = local + decay_t[..., None] * carry_in[:, :, None, :]
        return out.reshape(B, nb * C, D)[:, :M].to(x.dtype)

    def _ema(self, x: torch.Tensor, p: torch.Tensor, init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, M, D] chunk outputs (compacted); p: [B, M] boundary probs; init: [B, D] value carried in
        from earlier chunks (None = zero). Returns z̄ [B, M, D]."""
        if self.use_kernel and HAS_SSD_KERNEL and x.is_cuda and init is None:
            dt = torch.log(1 / (1 - p)).to(torch.bfloat16)  # so that exp(-dt) = 1 - p
            xs = (x / dt[..., None]).to(torch.bfloat16)
            A = -torch.ones(self.nheads, device=x.device, dtype=torch.float32)
            b = p.to(torch.bfloat16)
            c = torch.ones_like(b)
            out = mamba_chunk_scan_combined(
                xs.view(x.shape[0], x.shape[1], self.nheads, self.headdim),
                dt[..., None].expand(-1, -1, self.nheads),
                A,
                b[..., None, None],
                c[..., None, None],
                chunk_size=self.block_size,
            )
            return out.reshape(x.shape)
        return self._ema_chunked(x, p, init=init)

    def forward(
        self,
        hidden: torch.Tensor,  # [B, M, D] main-network outputs at boundary positions (compacted)
        boundary_mask: torch.Tensor,  # [B, L]
        boundary_prob: torch.Tensor,  # [B, L, 2]
        state: Optional[DeChunkState] = None,
    ) -> torch.Tensor:
        B, L = boundary_mask.shape
        p = torch.clamp(boundary_prob[..., 1].float(), min=self.eps, max=1 - self.eps)
        token_idx = torch.arange(L, device=hidden.device)[None, :] + (~boundary_mask).long() * L
        order = torch.argsort(token_idx, dim=1)[:, : hidden.shape[1]]
        p = torch.gather(p, 1, order)  # [B, M] probs of the selected positions
        if state is None:
            smoothed = self._ema(hidden, p)  # [B, M, D]
            plug = (torch.cumsum(boundary_mask.long(), dim=1) - 1).clamp(min=0)  # byte t -> its chunk index
        else:
            # continue from the carried value: bytes before this segment's first boundary keep it
            init = state.last_value.float()
            smoothed = self._ema(hidden, p, init=init) if hidden.shape[1] > 0 else hidden.float()[:, :0]
            smoothed = torch.cat([init[:, None, :], smoothed.float()], dim=1)  # slot 0 = carried value
            plug = torch.cumsum(boundary_mask.long(), dim=1)
        out = torch.gather(smoothed, 1, plug[:, :, None].expand(-1, -1, self.d_model))
        if state is not None:
            state.last_value.copy_(out[:, -1])
        return out.to(hidden.dtype)

    def step(self, hidden: torch.Tensor, boundary_mask: torch.Tensor, boundary_prob: torch.Tensor, state: DeChunkState) -> torch.Tensor:
        # hidden: [B', 1, D] for rows with a boundary; boundary_mask [B]; boundary_prob [B, 2]
        B = boundary_mask.shape[0]
        D = self.d_model
        p = torch.zeros(B, device=state.last_value.device, dtype=state.last_value.dtype)
        p[boundary_mask] = boundary_prob[boundary_mask, 1].clamp(min=self.eps, max=1 - self.eps).to(p.dtype)
        cur = torch.zeros(B, D, device=p.device, dtype=state.last_value.dtype)
        if hidden.shape[0] > 0:
            cur[boundary_mask] = hidden.squeeze(1).to(cur.dtype)
        result = p[:, None] * cur + (1 - p[:, None]) * state.last_value
        state.last_value.copy_(result)
        return result.unsqueeze(1)


# --------------------------------------------------------------------------------------
class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, g):
        return g


def ste_ones(x: torch.Tensor) -> torch.Tensor:
    """Forward: 1. Backward: identity. Used as out * STE(c_t) + residual."""
    return _STE.apply(x)


def ratio_loss(boundary_prob: torch.Tensor, boundary_mask: torch.Tensor, mask: torch.Tensor, target_ratio: float) -> torch.Tensor:
    """L_ratio = N/(N-1) * [(N-1) F G + (1-F)(1-G)],  F = mean(b_t), G = mean(p_t) over valid positions."""
    valid = mask.float()
    n = valid.sum().clamp(min=1.0)
    Fsel = (boundary_mask.float() * valid).sum() / n
    G = (boundary_prob[..., 1].float() * valid).sum() / n
    N = float(target_ratio)
    return (N / (N - 1.0)) * ((N - 1.0) * Fsel * G + (1.0 - Fsel) * (1.0 - G))


def atdc_target_ratio(step: int, total_steps: int, n_init: float, n_final: float, warmup_frac: float,
                      loss_window: Optional[float] = None, threshold: Optional[float] = None,
                      rate: float = 1.05) -> float:
    """ATDC's compression target at `step` (2605.30080, Alg. 1).

    Lines 4-9 are the schedule: hold N_init for `warmup_frac` of training, then ramp linearly to
    N_final. Mote shipped only that half until 2026-08-27, so N rose on wall-clock whether or not the
    model was coping with the compression it already had.

    Lines 10-15 are the trigger this adds: once a window of `adapt_window` steps has filled, N is
    boosted by `rate` (γ, the paper's 1.05) while the windowed mean LM loss sits under `threshold`
    (τ). The paper calls τ a "proficiency trigger" and sets it from a baseline run — the loss where
    the curve begins to plateau — with the warning that too high triggers premature compression and
    routing instability, too low never fires at all. `threshold=None` disables it, which is what
    every run before this date did implicitly.

    This moves the TARGET. It does not move the achieved rate by anything like the same amount: the
    ratio loss has no gradient path to the segmentation rate (see `ratio_loss`), and the paper's own
    ablation bought 0.57 BPIC for 2.0 of N. `project_boundaries` is the lever with a guarantee."""
    if total_steps <= 0:
        return n_init
    tw = int(total_steps * warmup_frac)
    if step < tw or total_steps <= tw:
        n = n_init
    else:
        rho = min((step - tw) / max(total_steps - tw, 1), 1.0)
        n = n_init + rho * (n_final - n_init)
    if threshold is not None and loss_window is not None and loss_window < threshold:
        n *= rate
    return n


def bytes_per_chunk(boundary_mask: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Empirical compression: valid bytes / boundaries (BPIC), as a 0-dim tensor so the training
    loop never synchronises on it (call float() at logging time)."""
    return mask.sum() / (boundary_mask & mask).sum().clamp(min=1)
