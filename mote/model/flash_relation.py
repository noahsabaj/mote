"""FlashRelation — fused Full Relation mixer in Triton, exact to the materialized form.

Forward (paper 2608.20172, Appendix A.3): the strictly-past Exchange entries E_ij = SiLU(U_ij)
are reduced with an online softmax (running max m, sum l, weighted sum z of Ĩ), never materializing
the T×T matrix. Self S_i = σ(U_ii/τ_S) stays out of the scan; the row is completed with
    L_i = m_i + log l_i,   Ī_i = z_i / l_i,   A_i = L_i − λ log i   (i one-based),
    g_i = σ(A_i − S_i),    Y_i = (1 − g_i) Ĩ_i + g_i Ī_i,
which equals softmax over R_i (F_ii = 1 − g_i and the −λ log i term cancels inside the past).

Backward (not in the paper; derived here):
    dĪ_i = g_i dY_i,  dg_i = dY_i·(Ī_i − Ĩ_i),  dA_i = dg_i g_i (1−g_i),  dS_i = −dA_i,
    dλ = −Σ_i dA_i log i,   dU_ii += dS_i S_i (1−S_i) / τ_S,
    w_ij = exp(E_ij − L_i),  Δ_i = dĪ_i·Ī_i,
    dE_ij = w_ij (dĪ_i·Ĩ_j − Δ_i + dA_i)          (the +dA_i is the logsumexp path through L_i)
    dU_ij = dE_ij SiLU'(U_ij),  dP1 = dU P2 /√d,  dP2 = dUᵀ P1 /√d,  dĨ_j = Σ_i w_ij dĪ_i + (1−g_j) dY_j.

v2 (2026-08-24, docs/results/2026-08-24-h100-probe.md):
* The backward is one pass parallel over key blocks, FlashAttention-2 style: each program holds a key /
  value block, walks the query blocks that can see it, recomputes U and w once, keeps dP2 and dĨ in
  registers and adds its dP1 partials into an fp32 buffer with atomics — 5 tile-dots per block pair
  instead of the 7 of the two-pass form, the exponentials once instead of twice. The per-row terms
  (dĪ, dA, Δ, dλ, the self-relation gradients) are a Triton prologue over query rows.
  `MOTE_DETERMINISTIC_RELATION=1` selects the two-pass kernels (no atomics, bitwise reproducible).
* Head dims that are not powers of two are tiled exactly as two power-of-two column blocks (96 = 64+32)
  instead of padding to the next power of two (a quarter of the MMA and shared-memory traffic at 96).
* Longest causal rows are scheduled first (LPT), and the online softmax only rescales its accumulators
  when a block raises the running max by more than log 256 (FlashAttention-4 §3.1.4); the true sum
  normalizes at the end. Exponentials are exp2 with the constants folded.
* Tiles are fixed per device (`_tiles`, `MOTE_RELATION_TILES` overrides), no runtime autotuning: with
  chunk-count bucketing the shapes repeat, and one compiled variant per head size keeps launch latency flat.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"
except Exception:  # pragma: no cover - Windows-native path
    triton = None
    tl = None
    HAS_TRITON = False

DETERMINISTIC = os.environ.get("MOTE_DETERMINISTIC_RELATION", "0") == "1"  # two-pass backward, no atomics

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
RESCALE_THRESH = 8.0  # log2(256): slack before the online softmax rescales (FlashAttention-4)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _split_d(D: int) -> Tuple[int, int, int, int, bool]:
    """Head dim D as up to two power-of-two column blocks: (D0, BLOCK_D0, D1, BLOCK_D1, HAS_D1).
    96 -> 64 + 32 exactly; 48 -> 32 + 16; 64 -> 64 + nothing; 112 -> 64 + 48 in a masked 64 block."""
    b0 = max(16, 1 << (D.bit_length() - 1))
    if b0 >= D:
        return D, b0, 0, 16, False
    d1 = D - b0
    return b0, b0, d1, max(16, _next_pow2(d1)), True


if HAS_TRITON:

    @triton.jit
    def _fwd_kernel(
        P1, P2, I, Y, IBAR, LSE, G, LAM,
        TQ, TK, sk, q_start, sm_scale, tau_inv,
        D: tl.constexpr, D0: tl.constexpr, BLOCK_D0: tl.constexpr, BLOCK_D1: tl.constexpr, HAS_D1: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, IEEE: tl.constexpr,
    ):
        # sk = row stride between heads of P2/I (TK*D when contiguous; the arena's capacity*D for a
        # prefix view of the decode arena, so no copy is made to read the cached chunks)
        pid_m = tl.num_programs(0) - 1 - tl.program_id(0)  # LPT: the longest causal rows first
        pid_bh = tl.program_id(1)
        sq = TQ * D
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d0 = tl.arange(0, BLOCK_D0)
        offs_d1 = D0 + tl.arange(0, BLOCK_D1)
        d0mask = offs_d0 < D0
        d1mask = offs_d1 < D
        mmask = offs_m < TQ
        qabs = offs_m + q_start
        qrow = P1 + pid_bh * sq + offs_m[:, None] * D
        q0 = tl.load(qrow + offs_d0[None, :], mask=mmask[:, None] & d0mask[None, :], other=0.0)
        if HAS_D1:
            q1 = tl.load(qrow + offs_d1[None, :], mask=mmask[:, None] & d1mask[None, :], other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc0 = tl.zeros([BLOCK_M, BLOCK_D0], tl.float32)
        acc1 = tl.zeros([BLOCK_M, BLOCK_D1], tl.float32)
        hi = tl.minimum(TK, (pid_m + 1) * BLOCK_M + q_start)
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            nmask = offs_n < TK
            krow = P2 + pid_bh * sk + offs_n[:, None] * D
            k0 = tl.load(krow + offs_d0[None, :], mask=nmask[:, None] & d0mask[None, :], other=0.0)
            if IEEE:
                u = tl.dot(q0, tl.trans(k0), input_precision="ieee")
            else:
                u = tl.dot(q0, tl.trans(k0))
            if HAS_D1:
                k1 = tl.load(krow + offs_d1[None, :], mask=nmask[:, None] & d1mask[None, :], other=0.0)
                if IEEE:
                    u += tl.dot(q1, tl.trans(k1), input_precision="ieee")
                else:
                    u += tl.dot(q1, tl.trans(k1))
            u = u * sm_scale
            e2 = (u * tl.sigmoid(u)) * LOG2E  # SiLU, in log2 units
            valid = (offs_n[None, :] < qabs[:, None]) & nmask[None, :]
            e2 = tl.where(valid, e2, float("-inf"))
            bmax = tl.max(e2, 1)
            need = bmax > m_i + RESCALE_THRESH  # rows whose max grew by more than 2^8; -inf rows never
            if tl.sum(need.to(tl.int32)) > 0:
                m_new = tl.where(need, bmax, m_i)
                m_new_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
                alpha = tl.exp2(m_i - m_new_safe)  # 1 where the max was kept, 0 for a fresh row
                acc0 = acc0 * alpha[:, None]
                if HAS_D1:
                    acc1 = acc1 * alpha[:, None]
                l_i = l_i * alpha
                m_i = m_new
            m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
            p = tl.exp2(e2 - m_safe[:, None])  # at most 2^8 between rescales
            l_i = l_i + tl.sum(p, 1)
            vrow = I + pid_bh * sk + offs_n[:, None] * D
            v0 = tl.load(vrow + offs_d0[None, :], mask=nmask[:, None] & d0mask[None, :], other=0.0)
            if IEEE:
                acc0 += tl.dot(p.to(v0.dtype), v0, input_precision="ieee")
            else:
                acc0 += tl.dot(p.to(v0.dtype), v0)
            if HAS_D1:
                v1 = tl.load(vrow + offs_d1[None, :], mask=nmask[:, None] & d1mask[None, :], other=0.0)
                if IEEE:
                    acc1 += tl.dot(p.to(v1.dtype), v1, input_precision="ieee")
                else:
                    acc1 += tl.dot(p.to(v1.dtype), v1)

        has = l_i > 0.0
        l_safe = tl.where(has, l_i, 1.0)
        lse = tl.where(has, (m_i + tl.log2(l_safe)) * LN2, float("-inf"))  # natural units
        ibar0 = acc0 / l_safe[:, None]
        ibar1 = acc1 / l_safe[:, None]
        # self relation on the diagonal: U_ii from the query's own key
        kqrow = P2 + pid_bh * sk + qabs[:, None] * D
        kq0 = tl.load(kqrow + offs_d0[None, :], mask=mmask[:, None] & d0mask[None, :], other=0.0)
        uii = tl.sum(q0.to(tl.float32) * kq0.to(tl.float32), 1)
        if HAS_D1:
            kq1 = tl.load(kqrow + offs_d1[None, :], mask=mmask[:, None] & d1mask[None, :], other=0.0)
            uii += tl.sum(q1.to(tl.float32) * kq1.to(tl.float32), 1)
        uii = uii * sm_scale
        s = tl.sigmoid(uii * tau_inv)
        lam = tl.load(LAM)
        a = lse - lam * tl.log((qabs + 1).to(tl.float32))
        g = tl.where(has, tl.sigmoid(a - s), 0.0)
        irow = I + pid_bh * sk + qabs[:, None] * D
        yrow = Y + pid_bh * sq + offs_m[:, None] * D
        brow = IBAR + pid_bh * sq + offs_m[:, None] * D
        iself0 = tl.load(irow + offs_d0[None, :], mask=mmask[:, None] & d0mask[None, :], other=0.0)
        y0 = (1.0 - g)[:, None] * iself0.to(tl.float32) + g[:, None] * ibar0
        omask0 = mmask[:, None] & d0mask[None, :]
        tl.store(yrow + offs_d0[None, :], y0.to(Y.dtype.element_ty), mask=omask0)
        tl.store(brow + offs_d0[None, :], ibar0, mask=omask0)
        if HAS_D1:
            iself1 = tl.load(irow + offs_d1[None, :], mask=mmask[:, None] & d1mask[None, :], other=0.0)
            y1 = (1.0 - g)[:, None] * iself1.to(tl.float32) + g[:, None] * ibar1
            omask1 = mmask[:, None] & d1mask[None, :]
            tl.store(yrow + offs_d1[None, :], y1.to(Y.dtype.element_ty), mask=omask1)
            tl.store(brow + offs_d1[None, :], ibar1, mask=omask1)
        tl.store(LSE + pid_bh * TQ + offs_m, lse, mask=mmask)
        tl.store(G + pid_bh * TQ + offs_m, g, mask=mmask)

    # ---- backward v2: row prologue + one pass over key blocks ------------------------------------------
    @triton.jit
    def _bwd_rows_kernel(
        P1, P2, I, DY, IBAR, G, DIBAR, DELTA, DA, DUII, DP1, DLAM,
        TQ, TK, q_start, sm_scale, tau_inv,
        D: tl.constexpr, D0: tl.constexpr, BLOCK_D0: tl.constexpr, BLOCK_D1: tl.constexpr, HAS_D1: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Per query row: dĪ = g dY (stored in the input dtype for the dots), Δ, dA, the self-relation
        gradient dU_ii (DUII) and its dP1 part (DP1 is initialised with it; the main kernel adds the past),
        and this block's share of dλ (one atomic add per program)."""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        sq = TQ * D
        sk = TK * D
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d0 = tl.arange(0, BLOCK_D0)
        offs_d1 = D0 + tl.arange(0, BLOCK_D1)
        d0mask = offs_d0 < D0
        d1mask = offs_d1 < D
        mmask = offs_m < TQ
        qabs = offs_m + q_start
        m0 = mmask[:, None] & d0mask[None, :]
        m1 = mmask[:, None] & d1mask[None, :]
        qoff = pid_bh * sq + offs_m[:, None] * D
        koff = pid_bh * sk + qabs[:, None] * D
        g = tl.load(G + pid_bh * TQ + offs_m, mask=mmask, other=0.0)
        dy0 = tl.load(DY + qoff + offs_d0[None, :], mask=m0, other=0.0).to(tl.float32)
        ibar0 = tl.load(IBAR + qoff + offs_d0[None, :], mask=m0, other=0.0)
        iself0 = tl.load(I + koff + offs_d0[None, :], mask=m0, other=0.0).to(tl.float32)
        dibar0 = g[:, None] * dy0
        tl.store(DIBAR + qoff + offs_d0[None, :], dibar0.to(DIBAR.dtype.element_ty), mask=m0)
        dgate = tl.sum(dy0 * (ibar0 - iself0), 1)
        delta = tl.sum(dibar0 * ibar0, 1)
        if HAS_D1:
            dy1 = tl.load(DY + qoff + offs_d1[None, :], mask=m1, other=0.0).to(tl.float32)
            ibar1 = tl.load(IBAR + qoff + offs_d1[None, :], mask=m1, other=0.0)
            iself1 = tl.load(I + koff + offs_d1[None, :], mask=m1, other=0.0).to(tl.float32)
            dibar1 = g[:, None] * dy1
            tl.store(DIBAR + qoff + offs_d1[None, :], dibar1.to(DIBAR.dtype.element_ty), mask=m1)
            dgate += tl.sum(dy1 * (ibar1 - iself1), 1)
            delta += tl.sum(dibar1 * ibar1, 1)
        da = dgate * g * (1.0 - g)
        tl.store(DELTA + pid_bh * TQ + offs_m, delta, mask=mmask)
        tl.store(DA + pid_bh * TQ + offs_m, da, mask=mmask)
        lg = tl.log((qabs + 1).to(tl.float32))
        tl.atomic_add(DLAM, -tl.sum(tl.where(mmask, da * lg, 0.0)))
        # self relation: U_ii = q_i·k_i/√d, S_i = σ(U_ii/τ), dS = −dA
        q0 = tl.load(P1 + qoff + offs_d0[None, :], mask=m0, other=0.0).to(tl.float32)
        kq0 = tl.load(P2 + koff + offs_d0[None, :], mask=m0, other=0.0).to(tl.float32)
        uii = tl.sum(q0 * kq0, 1)
        if HAS_D1:
            q1 = tl.load(P1 + qoff + offs_d1[None, :], mask=m1, other=0.0).to(tl.float32)
            kq1 = tl.load(P2 + koff + offs_d1[None, :], mask=m1, other=0.0).to(tl.float32)
            uii += tl.sum(q1 * kq1, 1)
        uii = uii * sm_scale
        s = tl.sigmoid(uii * tau_inv)
        duii = (-da) * s * (1.0 - s) * tau_inv
        tl.store(DUII + pid_bh * TQ + offs_m, duii, mask=mmask)
        tl.store(DP1 + qoff + offs_d0[None, :], duii[:, None] * kq0 * sm_scale, mask=m0)
        if HAS_D1:
            tl.store(DP1 + qoff + offs_d1[None, :], duii[:, None] * kq1 * sm_scale, mask=m1)

    @triton.jit
    def _bwd_kernel(
        P1, P2, I, DY, DIBAR, LSE, DELTA, DA, DUII, G, DP1, DP2, DI,
        TQ, TK, q_start, sm_scale,
        D: tl.constexpr, D0: tl.constexpr, BLOCK_D0: tl.constexpr, BLOCK_D1: tl.constexpr, HAS_D1: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, IEEE: tl.constexpr,
    ):
        """One key block per program: dP2 and dĨ accumulate in registers over the query blocks that can
        see it; dP1 partials are added into the fp32 DP1 buffer with atomics. Key rows that are also
        queries (n ≥ q_start) get their self terms — (1−g) dY into dĨ and dU_ii·P1 into dP2 — before the
        single store."""
        pid_n = tl.program_id(0)  # ascending = the key blocks with the most query blocks first
        pid_bh = tl.program_id(1)
        sq = TQ * D
        sk = TK * D
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d0 = tl.arange(0, BLOCK_D0)
        offs_d1 = D0 + tl.arange(0, BLOCK_D1)
        d0mask = offs_d0 < D0
        d1mask = offs_d1 < D
        nmask = offs_n < TK
        n0 = nmask[:, None] & d0mask[None, :]
        n1 = nmask[:, None] & d1mask[None, :]
        koff = pid_bh * sk + offs_n[:, None] * D
        k0 = tl.load(P2 + koff + offs_d0[None, :], mask=n0, other=0.0)
        v0 = tl.load(I + koff + offs_d0[None, :], mask=n0, other=0.0)
        if HAS_D1:
            k1 = tl.load(P2 + koff + offs_d1[None, :], mask=n1, other=0.0)
            v1 = tl.load(I + koff + offs_d1[None, :], mask=n1, other=0.0)
        acc_dk0 = tl.zeros([BLOCK_N, BLOCK_D0], tl.float32)
        acc_dv0 = tl.zeros([BLOCK_N, BLOCK_D0], tl.float32)
        acc_dk1 = tl.zeros([BLOCK_N, BLOCK_D1], tl.float32)
        acc_dv1 = tl.zeros([BLOCK_N, BLOCK_D1], tl.float32)
        # queries that can see this key block: qabs > pid_n*BLOCK_N  ->  local m > pid_n*BLOCK_N - q_start
        m_lo = tl.maximum(pid_n * BLOCK_N - q_start, 0)
        m_lo = (m_lo // BLOCK_M) * BLOCK_M
        for start_m in range(m_lo, TQ, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mmask = offs_m < TQ
            qabs = offs_m + q_start
            m0 = mmask[:, None] & d0mask[None, :]
            m1 = mmask[:, None] & d1mask[None, :]
            qoff = pid_bh * sq + offs_m[:, None] * D
            q0 = tl.load(P1 + qoff + offs_d0[None, :], mask=m0, other=0.0)
            do0 = tl.load(DIBAR + qoff + offs_d0[None, :], mask=m0, other=0.0)
            if IEEE:
                u = tl.dot(q0, tl.trans(k0), input_precision="ieee")
                dw = tl.dot(do0, tl.trans(v0), input_precision="ieee")
            else:
                u = tl.dot(q0, tl.trans(k0))
                dw = tl.dot(do0, tl.trans(v0))
            if HAS_D1:
                q1 = tl.load(P1 + qoff + offs_d1[None, :], mask=m1, other=0.0)
                do1 = tl.load(DIBAR + qoff + offs_d1[None, :], mask=m1, other=0.0)
                if IEEE:
                    u += tl.dot(q1, tl.trans(k1), input_precision="ieee")
                    dw += tl.dot(do1, tl.trans(v1), input_precision="ieee")
                else:
                    u += tl.dot(q1, tl.trans(k1))
                    dw += tl.dot(do1, tl.trans(v1))
            u = u * sm_scale
            lse = tl.load(LSE + pid_bh * TQ + offs_m, mask=mmask, other=0.0)
            delta = tl.load(DELTA + pid_bh * TQ + offs_m, mask=mmask, other=0.0)
            da = tl.load(DA + pid_bh * TQ + offs_m, mask=mmask, other=0.0)
            valid = (offs_n[None, :] < qabs[:, None]) & nmask[None, :] & mmask[:, None]
            s = tl.sigmoid(u)
            w = tl.where(valid, tl.exp2((u * s) * LOG2E - lse[:, None] * LOG2E), 0.0)
            de = w * (dw - delta[:, None] + da[:, None])
            du = de * (s * (1.0 + u * (1.0 - s)))  # SiLU'
            wt = tl.trans(w.to(do0.dtype))
            dut = tl.trans(du.to(q0.dtype))
            if IEEE:
                acc_dv0 += tl.dot(wt, do0, input_precision="ieee")
                acc_dk0 += tl.dot(dut, q0, input_precision="ieee")
                dq0 = tl.dot(du.to(k0.dtype), k0, input_precision="ieee")
            else:
                acc_dv0 += tl.dot(wt, do0)
                acc_dk0 += tl.dot(dut, q0)
                dq0 = tl.dot(du.to(k0.dtype), k0)
            tl.atomic_add(DP1 + qoff + offs_d0[None, :], dq0 * sm_scale, mask=m0, sem="relaxed")
            if HAS_D1:
                if IEEE:
                    acc_dv1 += tl.dot(wt, do1, input_precision="ieee")
                    acc_dk1 += tl.dot(dut, q1, input_precision="ieee")
                    dq1 = tl.dot(du.to(k1.dtype), k1, input_precision="ieee")
                else:
                    acc_dv1 += tl.dot(wt, do1)
                    acc_dk1 += tl.dot(dut, q1)
                    dq1 = tl.dot(du.to(k1.dtype), k1)
                tl.atomic_add(DP1 + qoff + offs_d1[None, :], dq1 * sm_scale, mask=m1, sem="relaxed")
        # self terms for the key rows that are also queries
        ms = offs_n - q_start
        smask = nmask & (ms >= 0)
        s0 = smask[:, None] & d0mask[None, :]
        s1 = smask[:, None] & d1mask[None, :]
        soff = pid_bh * sq + ms[:, None] * D
        g_s = tl.load(G + pid_bh * TQ + ms, mask=smask, other=0.0)
        duii_s = tl.load(DUII + pid_bh * TQ + ms, mask=smask, other=0.0)
        dy0 = tl.load(DY + soff + offs_d0[None, :], mask=s0, other=0.0).to(tl.float32)
        qs0 = tl.load(P1 + soff + offs_d0[None, :], mask=s0, other=0.0).to(tl.float32)
        acc_dv0 += (1.0 - g_s)[:, None] * dy0
        acc_dk0 += duii_s[:, None] * qs0
        tl.store(DP2 + koff + offs_d0[None, :], (acc_dk0 * sm_scale).to(DP2.dtype.element_ty), mask=n0)
        tl.store(DI + koff + offs_d0[None, :], acc_dv0.to(DI.dtype.element_ty), mask=n0)
        if HAS_D1:
            dy1 = tl.load(DY + soff + offs_d1[None, :], mask=s1, other=0.0).to(tl.float32)
            qs1 = tl.load(P1 + soff + offs_d1[None, :], mask=s1, other=0.0).to(tl.float32)
            acc_dv1 += (1.0 - g_s)[:, None] * dy1
            acc_dk1 += duii_s[:, None] * qs1
            tl.store(DP2 + koff + offs_d1[None, :], (acc_dk1 * sm_scale).to(DP2.dtype.element_ty), mask=n1)
            tl.store(DI + koff + offs_d1[None, :], acc_dv1.to(DI.dtype.element_ty), mask=n1)

    # ---- two-pass backward (deterministic fallback, the v1 kernels) ------------------------------------
    @triton.jit
    def _silu(u):
        return u * tl.sigmoid(u)

    @triton.jit
    def _dsilu(u):
        s = tl.sigmoid(u)
        return s * (1.0 + u * (1.0 - s))

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


# ---- tiles ------------------------------------------------------------------------------------------------
# (BLOCK_M, BLOCK_N, num_warps, num_stages) per (architecture, kind). Ada: 64×64 tiles fit the 99 KB of shared
# memory with the exact 64+32 head split at 2 stages. Hopper/Blackwell entries are the Ada values until the
# tile session on an H100 measures them (docs/shape.md, "Kernel and compile workstreams").
_TILES = {
    ("ada", "fwd"): (64, 64, 4, 2),
    ("ada", "bwd"): (64, 64, 4, 2),
    ("ada", "rows"): (64, 64, 4, 2),
    ("hopper", "fwd"): (64, 64, 4, 2),
    ("hopper", "bwd"): (64, 64, 4, 2),
    ("hopper", "rows"): (64, 64, 4, 2),
}
_ARCH_CACHE: dict = {}


def _arch(device) -> str:
    key = str(device)
    hit = _ARCH_CACHE.get(key)
    if hit is None:
        major = torch.cuda.get_device_capability(device)[0] if torch.cuda.is_available() and device.type == "cuda" else 8
        hit = "hopper" if major >= 9 else "ada"
        _ARCH_CACHE[key] = hit
    return hit


def _tiles(kind: str, device, itemsize: int) -> Tuple[int, int, int, int]:
    """Fixed tile config; `MOTE_RELATION_TILES="fwd:64,64,4,2;bwd:128,64,8,3"` overrides (the tuning session)."""
    env = os.environ.get("MOTE_RELATION_TILES")
    if env:
        for part in env.split(";"):
            k, _, v = part.partition(":")
            if k.strip() == kind:
                bm, bn, nw, ns = (int(x) for x in v.split(","))
                return bm, bn, nw, ns
    bm, bn, nw, ns = _TILES[(_arch(device), kind)]
    if itemsize == 4:  # fp32 operands (tests / reference runs): half the tile keeps the 4-byte tiles in shared memory
        bm, bn = min(bm, 32), min(bn, 32)
    return bm, bn, nw, ns


# ---- autograd through torch.library custom ops: one code path for eager and torch.compile (the ops are
# opaque to Dynamo, so a compiled Relation block has no graph break at the kernel) --------------------------
def _launch_fwd(p1, p2, info, lam, tau_s: float, q_start: int):
    # p1: [B,H,TQ,D] queries (absolute positions q_start..); p2, info: [B,H,TK,D] with TK = q_start + TQ
    B, H, TQ, D = p1.shape
    TK = p2.shape[2]
    assert TK == q_start + TQ, "keys must cover exactly the cached prefix plus the new positions"
    p1 = p1.contiguous()
    # Inference reads P2/I straight out of the decode arena (a [1,H,capacity,D] buffer sliced to TK
    # rows): rows are contiguous, heads are `capacity*D` apart. Pass that stride instead of copying.
    # The backward makes its own contiguous views, so this is safe under grad as well.
    strided_ok = (
        B == 1
        and p2.stride(3) == 1 and p2.stride(2) == D and info.stride(3) == 1 and info.stride(2) == D
        and p2.stride(1) == info.stride(1) and p2.stride(1) >= TK * D
    )
    if strided_ok:
        sk = p2.stride(1)
    else:
        p2, info = p2.contiguous(), info.contiguous()
        sk = TK * D
    lam32 = lam.detach().to(torch.float32).reshape(1).contiguous()
    y = torch.empty_like(p1, dtype=info.dtype)
    ibar = torch.empty(B, H, TQ, D, device=p1.device, dtype=torch.float32)
    lse = torch.empty(B, H, TQ, device=p1.device, dtype=torch.float32)
    g = torch.empty(B, H, TQ, device=p1.device, dtype=torch.float32)
    ieee = p1.dtype == torch.float32
    d0, bd0, d1, bd1, has_d1 = _split_d(D)
    bm, bn, nw, ns = _tiles("fwd", p1.device, p1.element_size())
    grid = (triton.cdiv(TQ, bm), B * H)
    _fwd_kernel[grid](
        p1, p2, info, y, ibar, lse, g, lam32,
        TQ, TK, sk, q_start, 1.0 / math.sqrt(D), 1.0 / tau_s,
        D=D, D0=d0, BLOCK_D0=bd0, BLOCK_D1=bd1, HAS_D1=has_d1,
        BLOCK_M=bm, BLOCK_N=bn, IEEE=ieee, num_warps=nw, num_stages=ns,
    )
    return y, g, ibar, lse


def _launch_bwd_one_pass(p1, p2, info, ibar, lse, g, dy, tau_s: float, q_start: int):
    B, H, TQ, D = p1.shape
    TK = p2.shape[2]
    scale = 1.0 / math.sqrt(D)
    ieee = p1.dtype == torch.float32
    p1, p2, info, dy = p1.contiguous(), p2.contiguous(), info.contiguous(), dy.contiguous()
    ibar, lse, g = ibar.contiguous(), lse.contiguous(), g.contiguous()
    dibar = torch.empty(B, H, TQ, D, device=p1.device, dtype=p1.dtype)
    delta = torch.empty(B, H, TQ, device=p1.device, dtype=torch.float32)
    da = torch.empty_like(delta)
    duii = torch.empty_like(delta)
    dp1 = torch.empty(B, H, TQ, D, device=p1.device, dtype=torch.float32)  # initialised by the row kernel
    dp2 = torch.empty(B, H, TK, D, device=p1.device, dtype=p2.dtype)
    di = torch.empty(B, H, TK, D, device=p1.device, dtype=info.dtype)
    dlam = torch.zeros(1, device=p1.device, dtype=torch.float32)
    d0, bd0, d1, bd1, has_d1 = _split_d(D)
    dims = dict(D=D, D0=d0, BLOCK_D0=bd0, BLOCK_D1=bd1, HAS_D1=has_d1)
    rm, _, rw, rs = _tiles("rows", p1.device, p1.element_size())
    _bwd_rows_kernel[(triton.cdiv(TQ, rm), B * H)](
        p1, p2, info, dy, ibar, g, dibar, delta, da, duii, dp1, dlam,
        TQ, TK, q_start, scale, 1.0 / tau_s, BLOCK_M=rm, num_warps=rw, num_stages=rs, **dims,
    )
    bm, bn, nw, ns = _tiles("bwd", p1.device, p1.element_size())
    _bwd_kernel[(triton.cdiv(TK, bn), B * H)](
        p1, p2, info, dy, dibar, lse, delta, da, duii, g, dp1, dp2, di,
        TQ, TK, q_start, scale, BLOCK_M=bm, BLOCK_N=bn, IEEE=ieee, num_warps=nw, num_stages=ns, **dims,
    )
    return dp1.to(p1.dtype), dp2, di, dlam.reshape(())


def _launch_bwd_two_pass(p1, p2, info, ibar, lse, g, dy, tau_s: float, q_start: int):
    """v1 backward: row terms in PyTorch, one kernel for dP2/dĨ and one for dP1 — no atomics."""
    B, H, TQ, D = p1.shape
    TK = p2.shape[2]
    scale = 1.0 / math.sqrt(D)
    ieee = p1.dtype == torch.float32
    p1, p2, info = p1.contiguous(), p2.contiguous(), info.contiguous()
    dy32 = dy.to(torch.float32)
    iself = info[:, :, q_start:].to(torch.float32)
    p2self = p2[:, :, q_start:]
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
    bd = _next_pow2(max(D, 16))
    bm = 64 if bd * p1.element_size() <= 256 else 32
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
    return dp1.to(p1.dtype), dp2.to(p2.dtype), di.to(info.dtype), dlam.to(torch.float32)


if HAS_TRITON:

    @torch.library.custom_op("mote::relation_fwd", mutates_args=())
    def _relation_fwd(p1: torch.Tensor, p2: torch.Tensor, info: torch.Tensor, lam: torch.Tensor, tau_s: float, q_start: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _launch_fwd(p1, p2, info, lam, tau_s, q_start)

    @_relation_fwd.register_fake
    def _relation_fwd_fake(p1, p2, info, lam, tau_s, q_start):
        B, H, TQ, D = p1.shape
        return (
            p1.new_empty((B, H, TQ, D), dtype=info.dtype),
            p1.new_empty((B, H, TQ), dtype=torch.float32),
            p1.new_empty((B, H, TQ, D), dtype=torch.float32),
            p1.new_empty((B, H, TQ), dtype=torch.float32),
        )

    @torch.library.custom_op("mote::relation_bwd", mutates_args=())
    def _relation_bwd(p1: torch.Tensor, p2: torch.Tensor, info: torch.Tensor, ibar: torch.Tensor, lse: torch.Tensor, g: torch.Tensor, dy: torch.Tensor, tau_s: float, q_start: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fn = _launch_bwd_two_pass if DETERMINISTIC else _launch_bwd_one_pass
        return fn(p1, p2, info, ibar, lse, g, dy, tau_s, q_start)

    @_relation_bwd.register_fake
    def _relation_bwd_fake(p1, p2, info, ibar, lse, g, dy, tau_s, q_start):
        return p1.new_empty(p1.shape), p2.new_empty(p2.shape), info.new_empty(info.shape), p1.new_empty((), dtype=torch.float32)

    def _setup_context(ctx, inputs, output):
        p1, p2, info, lam, tau_s, q_start = inputs
        y, g, ibar, lse = output
        ctx.save_for_backward(p1, p2, info, ibar, lse, g)
        ctx.tau_s, ctx.q_start = tau_s, q_start

    def _backward(ctx, dy, _dg, _dibar, _dlse):
        p1, p2, info, ibar, lse, g = ctx.saved_tensors
        dp1, dp2, di, dlam = _relation_bwd(p1, p2, info, ibar, lse, g, dy, ctx.tau_s, ctx.q_start)
        return dp1, dp2, di, dlam, None, None

    _relation_fwd.register_autograd(_backward, setup_context=_setup_context)


def flash_relation(p1: torch.Tensor, p2: torch.Tensor, info: torch.Tensor, lam: torch.Tensor, tau_s: float = 2.0, q_start: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Full Relation. p1 [B,H,T,D] (RoPE applied), p2/info [B,H,S+T,D] (RoPE / Givens applied).
    Returns (Y [B,H,T,D] in info's dtype, g [B,H,T] fp32 exchange mass; no gradient flows through g)."""
    if not HAS_TRITON or not (p1.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"):
        raise RuntimeError("flash_relation needs CUDA + Triton")
    y, g, _ibar, _lse = _relation_fwd(p1, p2, info, lam, float(tau_s), int(q_start))
    return y, g.detach()


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
