"""One fused Triton op for every contraction the spine does at byte resolution.

Measured motivation (Mote-138M, seq 8192, `--ckpt-main`, 2026-08-26): turning the spine on costs
+92 ms in the encoder and +95 ms in the decoder per forward, against a 289 ms baseline, and the main
network — which the spine does not touch — is unchanged. The per-op diff says where it goes:

    aten::bmm       0 -> 214 ms (800 calls)     the two einsums, lowered to a 4x4 batched matmul
    aten::copy_   122 -> 271 ms                 reshape/permute materialisations feeding that bmm
    elementwise     0 -> 253 ms                 the sigmoid/scale/add chain
    aten::add_      0 -> 115 ms
    aten::mul      71 -> 169 ms

flops/byte is 231.1 either way. Every millisecond of that is bandwidth and launch overhead on a
contraction whose inner dimension is 4 — exactly what a fused kernel exists to remove.

All four spine contractions are the same function:

    out[i, d] = sum_m H[i, m] * x[m, d]  +  p[i] * y[<i or nothing>, d]

    expand read   N_OUT=1  N_IN=n   H = h_pre[..., None, :]   no y     -> u [..., d]
    expand write  N_OUT=n  N_IN=n   H = h_res                 y bcast  -> x' [..., n, d]
    frac   read   N_OUT=n  N_IN=n   H = h_pre                 no y     -> u [..., n, d]
    frac   write  N_OUT=n  N_IN=n   H = h_res                 y per-i  -> x' [..., n, d]

One program per token, the whole stream width in registers: N_IN and N_OUT are 4, so the n-axis
loops unroll and the reductions that produce dH and dp close inside the program. No atomics, no
partial buffers, one pass over x forward and one over dout backward.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"
except Exception:  # pragma: no cover - no triton
    triton = None
    tl = None
    HAS_TRITON = False


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


if HAS_TRITON:

    @triton.jit
    def _mix_fwd_kernel(
        X, H, P, Y, OUT,
        D, sxt, sht, spt, syt, sot,
        N_IN: tl.constexpr, N_OUT: tl.constexpr, BLOCK_D: tl.constexpr,
        HAS_P: tl.constexpr, Y_PER_I: tl.constexpr,
    ):
        t = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        m_d = d < D
        # H is [N_OUT, N_IN] per token; N_OUT*N_IN is 1..16, so read it as scalars into registers.
        for i in tl.static_range(N_OUT):
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            for m in tl.static_range(N_IN):
                h = tl.load(H + t * sht + i * N_IN + m).to(tl.float32)
                x = tl.load(X + t * sxt + m * D + d, mask=m_d, other=0.0).to(tl.float32)
                acc += h * x
            if HAS_P:
                p = tl.load(P + t * spt + i).to(tl.float32)
                yo = i * D if Y_PER_I else 0
                # Y arrives in whatever the autocast dtype is; it is promoted in registers, so the
                # fp32 copy the wrapper used to materialise at every site never happens.
                y = tl.load(Y + t * syt + yo + d, mask=m_d, other=0.0).to(tl.float32)
                acc += p * y
            tl.store(OUT + t * sot + i * D + d, acc.to(OUT.dtype.element_ty), mask=m_d)

    @triton.jit
    def _mix_bwd_kernel(
        X, H, P, Y, DOUT, DX, DH, DP, DY,
        D, sxt, sht, spt, syt, sot,
        N_IN: tl.constexpr, N_OUT: tl.constexpr, BLOCK_D: tl.constexpr,
        HAS_P: tl.constexpr, Y_PER_I: tl.constexpr,
    ):
        t = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        m_d = d < D
        # dx[m] = sum_i H[i,m] * dout[i]; dH[i,m] = <dout[i], x[m]>; dp[i] = <dout[i], y[i]>;
        # dy = sum_i p[i]*dout[i]  (expand: y is shared, so the sum closes here) or p[i]*dout[i] (frac).
        for m in tl.static_range(N_IN):
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            x = tl.load(X + t * sxt + m * D + d, mask=m_d, other=0.0).to(tl.float32)
            for i in tl.static_range(N_OUT):
                do = tl.load(DOUT + t * sot + i * D + d, mask=m_d, other=0.0).to(tl.float32)
                h = tl.load(H + t * sht + i * N_IN + m).to(tl.float32)
                dx += h * do
                tl.store(DH + t * sht + i * N_IN + m, tl.sum(do * x, axis=0))
            # dx, dh and dp stay fp32: X is the residual and H/P are the spine's own coefficients.
            tl.store(DX + t * sxt + m * D + d, dx, mask=m_d)
        if HAS_P:
            dy_shared = tl.zeros([BLOCK_D], dtype=tl.float32)
            for i in tl.static_range(N_OUT):
                do = tl.load(DOUT + t * sot + i * D + d, mask=m_d, other=0.0).to(tl.float32)
                p = tl.load(P + t * spt + i).to(tl.float32)
                yo = i * D if Y_PER_I else 0
                y = tl.load(Y + t * syt + yo + d, mask=m_d, other=0.0).to(tl.float32)
                tl.store(DP + t * spt + i, tl.sum(do * y, axis=0))
                if Y_PER_I:
                    tl.store(DY + t * syt + i * D + d, (p * do).to(DY.dtype.element_ty), mask=m_d)
                else:
                    dy_shared += p * do
            if not Y_PER_I:
                tl.store(DY + t * syt + d, dy_shared.to(DY.dtype.element_ty), mask=m_d)


def _flat(t: Optional[torch.Tensor], nd: int) -> Optional[torch.Tensor]:
    """Collapse every leading axis into one token axis, keeping the last `nd` dims."""
    if t is None:
        return None
    return t.contiguous().reshape(-1, *t.shape[len(t.shape) - nd:])


def _launch_fwd(x, h, p, y, n_out: int, y_per_i: bool, out_dtype: torch.dtype):
    xf = _flat(x, 2)                      # [T, N_IN, D]
    T, n_in, D = xf.shape
    hf = _flat(h, 2)                      # [T, N_OUT, N_IN]
    pf = _flat(p, 1) if p is not None else None
    yf = _flat(y, 2 if y_per_i else 1) if y is not None else None
    out = torch.empty(T, n_out, D, device=xf.device, dtype=out_dtype)
    _mix_fwd_kernel[(T,)](
        xf, hf, pf if pf is not None else xf, yf if yf is not None else xf, out,
        D, n_in * D, n_out * n_in, n_out if pf is not None else 0,
        (n_out * D if y_per_i else D) if yf is not None else 0, n_out * D,
        N_IN=n_in, N_OUT=n_out, BLOCK_D=_next_pow2(D),
        HAS_P=pf is not None, Y_PER_I=y_per_i,
    )
    return out.reshape(*x.shape[:-2], n_out, D)


def _launch_bwd(x, h, p, y, dout, n_out: int, y_per_i: bool):
    xf, hf = _flat(x, 2), _flat(h, 2)
    T, n_in, D = xf.shape
    pf = _flat(p, 1) if p is not None else None
    yf = _flat(y, 2 if y_per_i else 1) if y is not None else None
    dof = _flat(dout, 2)
    dx = torch.empty_like(xf)
    dh = torch.empty_like(hf)
    dp = torch.empty_like(pf) if pf is not None else None
    dy = torch.empty_like(yf) if yf is not None else None
    _mix_bwd_kernel[(T,)](
        xf, hf, pf if pf is not None else xf, yf if yf is not None else xf, dof,
        dx, dh, dp if dp is not None else dx, dy if dy is not None else dx,
        D, n_in * D, n_out * n_in, n_out if pf is not None else 0,
        (n_out * D if y_per_i else D) if yf is not None else 0, n_out * D,
        N_IN=n_in, N_OUT=n_out, BLOCK_D=_next_pow2(D),
        HAS_P=pf is not None, Y_PER_I=y_per_i,
    )
    # Reshape here, not in the autograd hook: the fake kernel promises the caller's shapes, and a
    # real op that returned the flat token axis would only disagree with it under torch.compile.
    return (dx.reshape(x.shape), dh.reshape(h.shape),
            dp.reshape(p.shape) if dp is not None else None,
            dy.reshape(y.shape) if dy is not None else None)


if HAS_TRITON:

    @torch.library.custom_op("mote::spine_mix_fwd", mutates_args=())
    def _spine_mix_fwd(x: torch.Tensor, h: torch.Tensor, p: Optional[torch.Tensor],
                       y: Optional[torch.Tensor], n_out: int, y_per_i: bool,
                       out_dtype: torch.dtype) -> torch.Tensor:
        return _launch_fwd(x, h, p, y, n_out, y_per_i, out_dtype)

    @_spine_mix_fwd.register_fake
    def _spine_mix_fwd_fake(x, h, p, y, n_out, y_per_i, out_dtype):
        return x.new_empty((*x.shape[:-2], n_out, x.shape[-1]), dtype=out_dtype)

    @torch.library.custom_op("mote::spine_mix_bwd", mutates_args=())
    def _spine_mix_bwd(x: torch.Tensor, h: torch.Tensor, p: Optional[torch.Tensor],
                       y: Optional[torch.Tensor], dout: torch.Tensor, n_out: int,
                       y_per_i: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dx, dh, dp, dy = _launch_bwd(x, h, p, y, dout, n_out, y_per_i)
        # Distinct placeholders: a custom op's returns may not alias each other, so the two absent
        # gradients cannot be the same zero tensor.
        return (dx, dh,
                dp if dp is not None else x.new_zeros(()),
                dy if dy is not None else x.new_zeros(()))

    @_spine_mix_bwd.register_fake
    def _spine_mix_bwd_fake(x, h, p, y, dout, n_out, y_per_i):
        # dy carries y's own dtype: autograd requires each gradient to match its input.
        return (x.new_empty(x.shape, dtype=torch.float32), h.new_empty(h.shape, dtype=torch.float32),
                p.new_empty(p.shape, dtype=torch.float32) if p is not None else x.new_zeros(()),
                y.new_empty(y.shape) if y is not None else x.new_zeros(()))

    def _setup_context(ctx, inputs, output):
        x, h, p, y, n_out, y_per_i, _out_dtype = inputs
        ctx.save_for_backward(x, h, p, y)
        ctx.n_out, ctx.y_per_i = n_out, y_per_i
        ctx.has_p, ctx.has_y = p is not None, y is not None

    def _backward(ctx, dout):
        x, h, p, y = ctx.saved_tensors
        dx, dh, dp, dy = _spine_mix_bwd(x, h, p, y, dout.contiguous(), ctx.n_out, ctx.y_per_i)
        return dx, dh, dp if ctx.has_p else None, dy if ctx.has_y else None, None, None, None

    _spine_mix_fwd.register_autograd(_backward, setup_context=_setup_context)


def spine_mix(x: torch.Tensor, h: torch.Tensor, p: Optional[torch.Tensor] = None,
              y: Optional[torch.Tensor] = None, n_out: Optional[int] = None,
              y_per_i: bool = False, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """out[..., i, d] = sum_m h[..., i, m] * x[..., m, d] + p[..., i] * y[..., (i,) d].

    X, H and P are fp32 — X is the residual and H/P are the spine's own coefficients. `y` and the
    result carry whatever dtype the caller asks for: `write` returns the fp32 residual, `read`
    returns `u` in the sublayer's compute dtype. Both conversions happen in registers, so the
    fp32 round trip this used to force at every site is gone.
    """
    if n_out is None:
        n_out = h.shape[-2]
    if out_dtype is None:
        out_dtype = torch.float32
    if not HAS_TRITON or not (x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"):
        return spine_mix_reference(x, h, p, y, y_per_i, out_dtype)
    f = lambda t: None if t is None else t.float().contiguous()
    yc = None if y is None else y.contiguous()
    return _spine_mix_fwd(f(x), f(h), f(p), yc, int(n_out), bool(y_per_i), out_dtype)


def spine_mix_reference(x, h, p=None, y=None, y_per_i: bool = False,
                        out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """The materialised fp32 function the kernel implements (tests, and the CPU path)."""
    out = torch.einsum("...im,...md->...id", h.float(), x.float())
    if p is not None and y is not None:
        yf = y.float() if y_per_i else y.float().unsqueeze(-2)
        out = out + p.float().unsqueeze(-1) * yf
    return out if out_dtype is None else out.to(out_dtype)


# ---- the coefficient generator: RMSNorm over the flattened streams, then the phi projection -----
#
# `coefficients` reads the whole stream state to produce ~17 numbers per token. The RMSNorm writes
# a normalised [T, F] copy and the projection reads it straight back — at 16384 with n=4 expand,
# F is 2048 and that intermediate is 134 MB written and 134 MB read at each of seven sites, about
# 1.9 GB of traffic per forward for a result that is kilobytes. Fusing them means x is read once
# and the intermediate never exists.
#
# Only the forward is a kernel. The backward needs two reductions over the TOKEN axis (dphi and the
# norm's dw), which across programs would mean atomics and a non-deterministic result; instead it
# recomputes `normed` from x and the saved per-token rms — one pass — and lets cuBLAS do those two
# as ordinary GEMMs. Determinism is kept, and the memory that was saved is saved where it counts:
# forward no longer holds seven of those intermediates alive at once.


if HAS_TRITON:

    @triton.jit
    def _gen_proj_kernel(
        X, W, PHI, G, RMS,
        F, C, sxt, sgt,
        BLOCK_F: tl.constexpr, EPS: tl.constexpr,
    ):
        t = tl.program_id(0)
        f = tl.arange(0, BLOCK_F)
        m_f = f < F
        x = tl.load(X + t * sxt + f, mask=m_f, other=0.0).to(tl.float32)
        r = tl.rsqrt(tl.sum(x * x, axis=0) / F + EPS)
        tl.store(RMS + t, r)
        w = tl.load(W + f, mask=m_f, other=0.0).to(tl.float32)
        normed = x * r * w                      # never leaves registers
        for c in range(C):
            pw = tl.load(PHI + c * F + f, mask=m_f, other=0.0).to(tl.float32)
            tl.store(G + t * sgt + c, tl.sum(normed * pw, axis=0))


def _launch_gen_proj(x_flat, weight, phi_w, eps: float):
    T, F = x_flat.shape
    C = phi_w.shape[0]
    g = torch.empty(T, C, device=x_flat.device, dtype=torch.float32)
    rms = torch.empty(T, device=x_flat.device, dtype=torch.float32)
    _gen_proj_kernel[(T,)](x_flat, weight, phi_w, g, rms, F, C, F, C,
                           BLOCK_F=_next_pow2(F), EPS=eps)
    return g, rms


if HAS_TRITON:

    @torch.library.custom_op("mote::spine_gen_proj", mutates_args=())
    def _spine_gen_proj(x_flat: torch.Tensor, weight: torch.Tensor, phi_w: torch.Tensor,
                        eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        return _launch_gen_proj(x_flat, weight, phi_w, eps)

    @_spine_gen_proj.register_fake
    def _spine_gen_proj_fake(x_flat, weight, phi_w, eps):
        T = x_flat.shape[0]
        return (x_flat.new_empty((T, phi_w.shape[0]), dtype=torch.float32),
                x_flat.new_empty((T,), dtype=torch.float32))

    def _gp_setup(ctx, inputs, output):
        x_flat, weight, phi_w, eps = inputs
        ctx.save_for_backward(x_flat, weight, phi_w, output[1])
        ctx.eps = eps

    def _gp_backward(ctx, dg, _drms):
        x, w, phi_w, rms = ctx.saved_tensors
        F = x.shape[-1]
        dg = dg.contiguous().float()
        xr = x * rms.unsqueeze(-1)                       # x_hat, recomputed rather than stored
        normed = xr * w
        dphi_w = dg.transpose(0, 1) @ normed             # cuBLAS: deterministic, no atomics
        dnormed = dg @ phi_w
        dw = (dnormed * xr).sum(0)
        dxhat = dnormed * w
        # d/dx of x * rsqrt(mean(x^2) + eps), with the mean's own dependence on x
        dx = rms.unsqueeze(-1) * (dxhat - xr * (dxhat * xr).mean(-1, keepdim=True))
        return dx.to(x.dtype), dw.to(w.dtype), dphi_w.to(phi_w.dtype), None

    _spine_gen_proj.register_autograd(_gp_backward, setup_context=_gp_setup)


def gen_proj(x_flat: torch.Tensor, weight: torch.Tensor, phi_w: torch.Tensor,
             eps: float) -> torch.Tensor:
    """RMSNorm(x_flat) @ phi_w.T, without ever materialising the normalised intermediate."""
    if not HAS_TRITON or not (x_flat.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"):
        return gen_proj_reference(x_flat, weight, phi_w, eps)
    lead, F = x_flat.shape[:-1], x_flat.shape[-1]
    g, _rms = _spine_gen_proj(x_flat.contiguous().reshape(-1, F).float(),
                              weight.float(), phi_w.float(), float(eps))
    return g.reshape(*lead, phi_w.shape[0])


def gen_proj_reference(x_flat: torch.Tensor, weight: torch.Tensor, phi_w: torch.Tensor,
                       eps: float) -> torch.Tensor:
    """The materialised function: what RMSNorm followed by nn.Linear(bias=False) computes."""
    xf = x_flat.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()) @ phi_w.float().transpose(0, 1)
