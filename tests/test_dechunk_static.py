"""The dechunk EMA's closed-form cross-block carry equals the sequential block loop it replaced (values
and gradients, with and without a carried-in value, M not a multiple of the block), and the whole
EMA equals a plain sequential recurrence."""

import torch
import torch.nn.functional as F

from mote.model.dc import DeChunkLayer


def _ema_loop_reference(x, p, C=64, init=None):
    """The pre-2026-08-24 implementation: within-block closed form, Python loop across blocks."""
    B, M, D = x.shape
    wdt = torch.float64 if x.dtype == torch.float64 else torch.float32
    xf, pf = x.to(wdt), p.to(wdt)
    pad = (-M) % C
    if pad:
        xf = F.pad(xf, (0, 0, 0, pad))
        pf = F.pad(pf, (0, pad))
    nb = xf.shape[1] // C
    xf = xf.view(B, nb, C, D)
    pf = pf.view(B, nb, C)
    logq = torch.log1p(-pf)
    L = torch.cumsum(logq, dim=-1)
    diff = L[..., :, None] - L[..., None, :]
    mask = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.bool))
    A = torch.exp(diff.masked_fill(~mask, float("-inf")))
    local = torch.matmul(A, pf[..., None] * xf)
    carry = torch.zeros(B, D, device=x.device, dtype=wdt) if init is None else init.to(wdt)
    decay_t = torch.exp(L)
    blocks = []
    for b in range(nb):
        blk = local[:, b] + decay_t[:, b, :, None] * carry[:, None, :]
        blocks.append(blk)
        carry = blk[:, -1]
    out = torch.stack(blocks, dim=1)
    return out.reshape(B, nb * C, D)[:, :M].to(x.dtype)


def _ema_sequential(x, p, init=None):
    B, M, D = x.shape
    z = torch.zeros(B, D, dtype=x.dtype) if init is None else init.to(x.dtype)
    outs = []
    for t in range(M):
        z = p[:, t, None] * x[:, t] + (1 - p[:, t, None]) * z
        outs.append(z)
    return torch.stack(outs, dim=1)


@torch.no_grad()
def test_closed_form_matches_loop_and_sequential():
    torch.manual_seed(0)
    for M, C, with_init in [(300, 64, False), (300, 64, True), (64, 64, True), (5, 64, False), (257, 32, True)]:
        x = torch.randn(2, M, 8, dtype=torch.float64)
        p = torch.rand(2, M, dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
        init = torch.randn(2, 8, dtype=torch.float64) if with_init else None
        a = DeChunkLayer._ema_chunked(x, p, C=C, init=init)
        b = _ema_loop_reference(x, p, C=C, init=init)
        c = _ema_sequential(x, p, init=init)
        assert torch.allclose(a, b, atol=1e-10, rtol=1e-10), (M, C, with_init, (a - b).abs().max())
        assert torch.allclose(a, c, atol=1e-9, rtol=1e-9), (M, C, with_init, (a - c).abs().max())


def test_closed_form_gradients_match_loop():
    torch.manual_seed(1)
    x = torch.randn(2, 200, 8, dtype=torch.float64)
    p = torch.rand(2, 200, dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
    init = torch.randn(2, 8, dtype=torch.float64)
    dy = torch.randn(2, 200, 8, dtype=torch.float64)
    grads = []
    for fn in (DeChunkLayer._ema_chunked, _ema_loop_reference):
        xa, pa, ia = x.clone().requires_grad_(True), p.clone().requires_grad_(True), init.clone().requires_grad_(True)
        (fn(xa, pa, C=64, init=ia) * dy).sum().backward()
        grads.append((xa.grad, pa.grad, ia.grad))
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-9, rtol=1e-9), (a - b).abs().max()
