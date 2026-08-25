"""The batched Newton-Schulz Muon step equals the per-parameter reference step (the pre-2026-08-24
implementation) on every matrix, including a mixed shape group and Muon-SW's scaled decay."""

import copy

import torch

from mote.train.muon import Muon


@torch.no_grad()
def _ns_ref(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    X = (X / (X.norm() + eps)).to(torch.bfloat16)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


@torch.no_grad()
def _ref_step(params, grads, bufs, lr, mom, wd, nesterov, rms_scale, sw_decay, lr_max):
    for p, g, buf in zip(params, grads, bufs):
        buf.mul_(mom).add_(g)
        upd = g.add(buf, alpha=mom) if nesterov else buf
        upd = _ns_ref(upd).to(p.dtype)
        upd = upd * (rms_scale * max(p.shape[0], p.shape[1]) ** 0.5)
        if wd:
            decay = lr * lr / lr_max if sw_decay else lr
            p.mul_(1.0 - decay * wd)
        p.add_(upd, alpha=-lr)


import pytest

import mote.train.muon as muon_mod


@pytest.mark.parametrize("stack_cap", [None, 64 * 48 * 4 * 2, 1])  # unbounded / two matrices per group / one
def test_batched_step_matches_reference(stack_cap, monkeypatch):
    if stack_cap is not None:
        monkeypatch.setattr(muon_mod, "MAX_STACK_BYTES", stack_cap)
    torch.manual_seed(0)
    shapes = [(64, 48), (64, 48), (64, 48), (32, 96), (40, 40), (32, 96)]
    params = [torch.nn.Parameter(torch.randn(*s) * 0.1) for s in shapes]
    grads = [torch.randn(*s) * 0.01 for s in shapes]
    ref_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    ref_bufs = [torch.zeros_like(p) for p in params]
    for sw in (False, True):
        ps = [torch.nn.Parameter(p.detach().clone()) for p in params]
        rp = [torch.nn.Parameter(p.detach().clone()) for p in ref_params]
        rb = [b.clone() for b in ref_bufs]
        opt = Muon(ps, lr=1e-2, momentum=0.95, nesterov=True, weight_decay=0.1, sw_decay=sw, lr_max=2e-2)
        for it in range(3):  # momentum buffers evolve
            for p, g in zip(ps, grads):
                p.grad = g * (1.0 + 0.1 * it)
            opt.step()
            _ref_step(rp, [g * (1.0 + 0.1 * it) for g in grads], rb, 1e-2, 0.95, 0.1, True, 0.2, sw, 2e-2)
        for a, b in zip(ps, rp):
            # bf16 iterations in a batched matmul may round differently; the orthogonal factor agrees closely
            assert torch.allclose(a, b, atol=2e-3, rtol=1e-2), (a - b).abs().max().item()
