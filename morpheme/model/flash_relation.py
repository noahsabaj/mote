"""FlashRelation — fused Full Relation mixer in Triton, exact to the materialized form.

Forward (paper 2608.20172, Appendix A.3): the strictly-past Exchange entries E_ij = SiLU(U_ij)
are reduced with an online softmax (running max m, sum l, weighted sum z of Ĩ), never materializing
the T×T matrix. Self S_i = σ(U_ii/τ_S) stays out of the scan; the row is completed with
    L_i = m_i + log l_i,   Ī_i = z_i / l_i,   A_i = L_i − λ log i   (i one-based),
    g_i = σ(A_i − S_i),    Y_i = (1 − g_i) Ĩ_i + g_i Ī_i,
which equals softmax over R_i (F_ii = 1 − g_i and the −λ log i term cancels inside the past).

Backward (not in the paper; derived here, FlashAttention-2 style with tiles recomputed):
    dĪ_i = g_i dY_i,  dg_i = dY_i·(Ī_i − Ĩ_i),  dA_i = dg_i g_i (1−g_i),  dS_i = −dA_i,
    dλ = −Σ_i dA_i log i,   dU_ii += dS_i S_i (1−S_i) / τ_S,
    w_ij = exp(E_ij − L_i),  Δ_i = dĪ_i·Ī_i,
    dE_ij = w_ij (dĪ_i·Ĩ_j − Δ_i + dA_i)          (the +dA_i is the logsumexp path through L_i)
    dU_ij = dE_ij SiLU'(U_ij),  dP1 = dU P2 /√d,  dP2 = dUᵀ P1 /√d,  dĨ_j = Σ_i w_ij dĪ_i + (1−g_j) dY_j.
Fixed tile sizes (no autotuning): with chunk-count bucketing the shapes repeat, and a single
compiled variant per head size is what keeps launch latency flat.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = torch.cuda.is_available()
except Exception:  # pragma: no cover - Windows-native path
    triton = None
    tl = None
    HAS_TRITON = False


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


if HAS_TRITON:

    @triton.jit
    def _silu(u):
        return u * tl.sigmoid(u)

    @triton.jit
    def _dsilu(u):
        s = tl.sigmoid(u)
        return s * (1.0 + u * (1.0 - s))

    @triton.jit
    def _fwd_kernel(
        P1, P2, I, Y, IBAR, LSE, G, LAM,
        TQ, TK, q_start, sm_scale, tau_inv,
        D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, IEEE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        sq = TQ * D
        sk = TK * D
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mmask = offs_m < TQ
        dmask = offs_d < D
        qabs = offs_m + q_start
        q = tl.load(P1 + pid_bh * sq + offs_m[:, None] * D + offs_d[None, :], mask=mmask[:, None] & dmask[None, :], other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        hi = tl.minimum(TK, (pid_m + 1) * BLOCK_M + q_start)
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            nmask = offs_n < TK
            k = tl.load(P2 + pid_bh * sk + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None] & dmask[None, :], other=0.0)
            if IEEE:
                u = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
            else:
                u = tl.dot(q, tl.trans(k)) * sm_scale
            e = _silu(u)
            valid = (offs_n[None, :] < qabs[:, None]) & nmask[None, :]
            e = tl.where(valid, e, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(e, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(e - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(I + pid_bh * sk + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None] & dmask[None, :], other=0.0)
            if IEEE:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        has = l_i > 0.0
        l_safe = tl.where(has, l_i, 1.0)
        lse = tl.where(has, m_i + tl.log(l_safe), float("-inf"))
        ibar = acc / l_safe[:, None]
        # self relation on the diagonal: U_ii from the query's own key
        kq = tl.load(P2 + pid_bh * sk + qabs[:, None] * D + offs_d[None, :], mask=mmask[:, None] & dmask[None, :], other=0.0)
        uii = tl.sum(q.to(tl.float32) * kq.to(tl.float32), 1) * sm_scale
        s = tl.sigmoid(uii * tau_inv)
        lam = tl.load(LAM)
        a = lse - lam * tl.log((qabs + 1).to(tl.float32))
        g = tl.where(has, tl.sigmoid(a - s), 0.0)
        iself = tl.load(I + pid_bh * sk + qabs[:, None] * D + offs_d[None, :], mask=mmask[:, None] & dmask[None, :], other=0.0)
        y = (1.0 - g)[:, None] * iself.to(tl.float32) + g[:, None] * ibar

        omask = mmask[:, None] & dmask[None, :]
        tl.store(Y + pid_bh * sq + offs_m[:, None] * D + offs_d[None, :], y.to(Y.dtype.element_ty), mask=omask)
        tl.store(IBAR + pid_bh * sq + offs_m[:, None] * D + offs_d[None, :], ibar, mask=omask)
        tl.store(LSE + pid_bh * TQ + offs_m, lse, mask=mmask)
        tl.store(G + pid_bh * TQ + offs_m, g, mask=mmask)

    @triton.jit
    def _bwd_kv_kernel(
        P1, P2, I, DIBAR, LSE, DELTA, DA, DP2, DI,
        TQ, TK, q_start, sm_scale,
        D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, IEEE: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)
        sq = TQ * D
        sk = TK * D
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        nmask = offs_n < TK
        dmask = offs_d < D
        k = tl.load(P2 + pid_bh * sk + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None] & dmask[None, :], other=0.0)
        v = tl.load(I + pid_bh * sk + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None] & dmask[None, :], other=0.0)
        acc_dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
        acc_dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
        # queries that can see this key block: qabs > pid_n*BLOCK_N  ->  local m > pid_n*BLOCK_N - q_start
        m0 = tl.maximum(pid_n * BLOCK_N - q_start, 0)
        m0 = (m0 // BLOCK_M) * BLOCK_M
        for start_m in range(m0, TQ, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mmask = offs_m < TQ
            qabs = offs_m + q_start
            q = tl.load(P1 + pid_bh * sq + offs_m[:, None] * D + offs_d[None, :], mask=mmask[:, None] & dmask[None, :], other=0.0)
            do = tl.load(DIBAR + pid_bh * sq + offs_m[:, None] * D + offs_d[None, :], mask=mmask[:, None] & dmask[None, :], other=0.0)
            lse = tl.load(LSE + pid_bh * TQ + offs_m, mask=mmask, other=0.0)
            delta = tl.load(DELTA + pid_bh * TQ + offs_m, mask=mmask, other=0.0)
            da = tl.load(DA + pid_bh * TQ + offs_m, mask=mmask, other=0.0)
            if IEEE:
                u = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
                dw = tl.dot(do, tl.trans(v), input_precision="ieee")
            else:
                u = tl.dot(q, tl.trans(k)) * sm_scale
                dw = tl.dot(do, tl.trans(v))
            valid = (offs_n[None, :] < qabs[:, None]) & nmask[None, :] & mmask[:, None]
            w = tl.where(valid, tl.exp(_silu(u) - lse[:, None]), 0.0)
            de = w * (dw - delta[:, None] + da[:, None])
            du = de * _dsilu(u)
            if IEEE:
                acc_dv += tl.dot(tl.trans(w.to(do.dtype)), do, input_precision="ieee")
                acc_dk += tl.dot(tl.trans(du.to(q.dtype)), q, input_precision="ieee")
            else:
                acc_dv += tl.dot(tl.trans(w.to(do.dtype)), do)
                acc_dk += tl.dot(tl.trans(du.to(q.dtype)), q)
        omask = nmask[:, None] & dmask[None, :]
        tl.store(DP2 + pid_bh * sk + offs_n[:, None] * D + offs_d[None, :], acc_dk * sm_scale, mask=omask)
        tl.store(DI + pid_bh * sk + offs_n[:, None] * D + offs_d[None, :], acc_dv, mask=omask)

    @triton.jit
    def _bwd_q_kernel(
        P1, P2, I, DIBAR, LSE, DELTA, DA, DP1,
        TQ, TK, q_start, sm_scale,
        D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, IEEE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        sq = TQ * D
        sk = TK * D
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mmask = offs_m < TQ
        dmask = offs_d < D
        qabs = offs_m + q_start
        q = tl.load(P1 + pid_bh * sq + offs_m[:, None] * D + offs_d[None, :], mask=mmask[:, None] & dmask[None, :], other=0.0)
        do = tl.load(DIBAR + pid_bh * sq + offs_m[:, None] * D + offs_d[None, :], mask=mmask[:, None] & dmask[None, :], other=0.0)
        lse = tl.load(LSE + pid_bh * TQ + offs_m, mask=mmask, other=0.0)
        delta = tl.load(DELTA + pid_bh * TQ + offs_m, mask=mmask, other=0.0)
        da = tl.load(DA + pid_bh * TQ + offs_m, mask=mmask, other=0.0)
        acc_dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        hi = tl.minimum(TK, (pid_m + 1) * BLOCK_M + q_start)
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            nmask = offs_n < TK
            k = tl.load(P2 + pid_bh * sk + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None] & dmask[None, :], other=0.0)
            v = tl.load(I + pid_bh * sk + offs_n[:, None] * D + offs_d[None, :], mask=nmask[:, None] & dmask[None, :], other=0.0)
            if IEEE:
                u = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
                dw = tl.dot(do, tl.trans(v), input_precision="ieee")
            else:
                u = tl.dot(q, tl.trans(k)) * sm_scale
                dw = tl.dot(do, tl.trans(v))
            valid = (offs_n[None, :] < qabs[:, None]) & nmask[None, :] & mmask[:, None]
            w = tl.where(valid, tl.exp(_silu(u) - lse[:, None]), 0.0)
            de = w * (dw - delta[:, None] + da[:, None])
            du = de * _dsilu(u)
            if IEEE:
                acc_dq += tl.dot(du.to(k.dtype), k, input_precision="ieee")
            else:
                acc_dq += tl.dot(du.to(k.dtype), k)
        tl.store(DP1 + pid_bh * sq + offs_m[:, None] * D + offs_d[None, :], acc_dq * sm_scale, mask=mmask[:, None] & dmask[None, :])


def _tile(t: torch.Tensor, block_d: int) -> int:
    """64×64 tiles fit Ada's 99 KB of shared memory up to 128-wide bf16 heads; wider/fp32 tiles drop to 32."""
    return 64 if block_d * t.element_size() <= 256 else 32


class _FlashRelationFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p1, p2, info, lam, tau_s: float, q_start: int):
        # p1: [B,H,TQ,D] queries (absolute positions q_start..); p2, info: [B,H,TK,D] with TK = q_start + TQ
        B, H, TQ, D = p1.shape
        TK = p2.shape[2]
        assert TK == q_start + TQ, "keys must cover exactly the cached prefix plus the new positions"
        p1, p2, info = p1.contiguous(), p2.contiguous(), info.contiguous()
        lam32 = lam.detach().to(torch.float32).reshape(1).contiguous()
        y = torch.empty_like(p1, dtype=info.dtype)
        ibar = torch.empty(B, H, TQ, D, device=p1.device, dtype=torch.float32)
        lse = torch.empty(B, H, TQ, device=p1.device, dtype=torch.float32)
        g = torch.empty(B, H, TQ, device=p1.device, dtype=torch.float32)
        ieee = p1.dtype == torch.float32
        bd = _next_pow2(max(D, 16))
        bm = _tile(p1, bd)
        grid = (triton.cdiv(TQ, bm), B * H)
        _fwd_kernel[grid](
            p1, p2, info, y, ibar, lse, g, lam32,
            TQ, TK, q_start, 1.0 / math.sqrt(D), 1.0 / tau_s,
            D=D, BLOCK_D=bd, BLOCK_M=bm, BLOCK_N=bm, IEEE=ieee,
            num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(p1, p2, info, ibar, lse, g)
        ctx.tau_s, ctx.q_start, ctx.ieee, ctx.tile = tau_s, q_start, ieee, (bd, bm)
        ctx.mark_non_differentiable(g)
        return y, g

    @staticmethod
    def backward(ctx, dy, _dg):
        p1, p2, info, ibar, lse, g = ctx.saved_tensors
        tau_s, q_start, ieee = ctx.tau_s, ctx.q_start, ctx.ieee
        B, H, TQ, D = p1.shape
        TK = p2.shape[2]
        scale = 1.0 / math.sqrt(D)
        dy32 = dy.to(torch.float32)
        iself = info[:, :, q_start:].to(torch.float32)
        p2self = p2[:, :, q_start:]
        # row-level terms (fp32)
        g_ = g[..., None]
        dibar32 = g_ * dy32
        di_direct = (1.0 - g_) * dy32
        dgate = (dy32 * (ibar - iself)).sum(-1)
        da = dgate * g * (1.0 - g)
        delta = (dibar32 * ibar).sum(-1)
        qabs = torch.arange(q_start, q_start + TQ, device=p1.device, dtype=torch.float32)
        dlam = -(da * torch.log(qabs + 1.0)).sum()
        uii = (p1.to(torch.float32) * p2self.to(torch.float32)).sum(-1) * scale
        s = torch.sigmoid(uii / tau_s)
        duii = (-da) * s * (1.0 - s) / tau_s  # dS = -dA
        dp1_self = duii[..., None] * p2self.to(torch.float32) * scale
        dp2_self = duii[..., None] * p1.to(torch.float32) * scale

        dibar = dibar32.to(p1.dtype).contiguous()
        dp1 = torch.empty(B, H, TQ, D, device=p1.device, dtype=torch.float32)
        dp2 = torch.empty(B, H, TK, D, device=p1.device, dtype=torch.float32)
        di = torch.empty(B, H, TK, D, device=p1.device, dtype=torch.float32)
        lse_c, delta_c, da_c = lse.contiguous(), delta.contiguous(), da.contiguous()
        bd, bm = ctx.tile
        common = dict(D=D, BLOCK_D=bd, BLOCK_M=bm, BLOCK_N=bm, IEEE=ieee, num_warps=4, num_stages=2)
        _bwd_kv_kernel[(triton.cdiv(TK, bm), B * H)](
            p1, p2, info, dibar, lse_c, delta_c, da_c, dp2, di, TQ, TK, q_start, scale, **common
        )
        _bwd_q_kernel[(triton.cdiv(TQ, bm), B * H)](
            p1, p2, info, dibar, lse_c, delta_c, da_c, dp1, TQ, TK, q_start, scale, **common
        )
        dp1 = dp1 + dp1_self
        dp2[:, :, q_start:] += dp2_self
        di[:, :, q_start:] += di_direct
        return dp1.to(p1.dtype), dp2.to(p2.dtype), di.to(info.dtype), dlam.to(torch.float32), None, None


def flash_relation(p1: torch.Tensor, p2: torch.Tensor, info: torch.Tensor, lam: torch.Tensor, tau_s: float = 2.0, q_start: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Full Relation. p1 [B,H,T,D] (RoPE applied), p2/info [B,H,S+T,D] (RoPE / Givens applied).
    Returns (Y [B,H,T,D] in info's dtype, g [B,H,T] fp32 exchange mass)."""
    if not HAS_TRITON or not p1.is_cuda:
        raise RuntimeError("flash_relation needs CUDA + Triton")
    return _FlashRelationFn.apply(p1, p2, info, lam, float(tau_s), int(q_start))


def relation_reference(p1, p2, info, lam, tau_s: float = 2.0, q_start: int = 0):
    """Materialized fp32 reference of the same function (for tests)."""
    B, H, TQ, D = p1.shape
    TK = p2.shape[2]
    p1f, p2f, inff = p1.float(), p2.float(), info.float()
    u = torch.matmul(p1f, p2f.transpose(-1, -2)) / math.sqrt(D)
    qi = torch.arange(q_start, q_start + TQ, device=p1.device)
    kj = torch.arange(0, TK, device=p1.device)
    is_self = qi[:, None] == kj[None, :]
    is_past = kj[None, :] < qi[:, None]
    log_i = torch.log((qi + 1).float())[:, None]
    r = torch.where(is_self, torch.sigmoid(u / tau_s), torch.nn.functional.silu(u) - lam.float() * log_i)
    r = r.masked_fill(~(is_self | is_past), float("-inf"))
    flow = torch.softmax(r, dim=-1)
    y = torch.matmul(flow, inff)
    g = flow.masked_fill(~is_past, 0.0).sum(-1)
    return y, g
