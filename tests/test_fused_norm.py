"""The fused Triton add+RMSNorm (phase K, docs/shape.md) must match the pure-PyTorch reference in
forward values, the fp32 residual stream, and every gradient — the flagship trains on it."""

import os

import pytest
import torch

from mote.model.norm import HAS_FUSED_NORM, RMSNorm

needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_FUSED_NORM), reason="needs CUDA + the fused kernel"
)


def _pair(D=384, seed=0):
    torch.manual_seed(seed)
    ref = RMSNorm(D, device="cuda", dtype=torch.float32)
    fused = RMSNorm(D, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        w = torch.randn(D, device="cuda") * 0.2 + 1.0
        ref.weight.copy_(w)
        fused.weight.copy_(w)
    return ref, fused


def _run(m, x, res, prenorm, fused_on, retain_res):
    os.environ.pop("MOTE_NO_FUSED_NORM", None)
    if not fused_on:
        os.environ["MOTE_NO_FUSED_NORM"] = "1"
    x = x.clone().requires_grad_(True)
    res_in = res.clone().requires_grad_(True) if res is not None else None
    try:
        out = m(x, residual=res_in, prenorm=prenorm)
    finally:
        os.environ.pop("MOTE_NO_FUSED_NORM", None)
    if prenorm:
        y, new_res = out
        if retain_res:
            new_res.retain_grad()
        loss = y.float().pow(2).mean() + new_res.float().sin().mean()
    else:
        y, new_res = out, None
        loss = y.float().pow(2).mean()
    loss.backward()
    return y, new_res, x.grad, (res_in.grad if res_in is not None else None), m.weight.grad


@needs_gpu
@pytest.mark.parametrize("with_residual", [True, False])
def test_fused_matches_reference_prenorm(with_residual):
    ref, fused = _pair()
    x = torch.randn(2, 96, 384, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(2, 96, 384, device="cuda", dtype=torch.float32) if with_residual else None
    y_r, res_r, gx_r, gres_r, gw_r = _run(ref, x, res, True, fused_on=False, retain_res=False)
    y_f, res_f, gx_f, gres_f, gw_f = _run(fused, x, res, True, fused_on=True, retain_res=False)
    assert y_f.dtype == y_r.dtype == torch.bfloat16
    assert res_f.dtype == res_r.dtype == torch.float32  # the residual stream stays fp32
    assert torch.allclose(y_f.float(), y_r.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(res_f, res_r, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gx_f.float(), gx_r.float(), atol=2e-2, rtol=2e-2)
    if with_residual:
        assert torch.allclose(gres_f, gres_r, atol=1e-3, rtol=1e-3)
    assert torch.allclose(gw_f, gw_r, atol=1e-2, rtol=1e-2)


@needs_gpu
def test_fused_matches_reference_final_norm():
    ref, fused = _pair(seed=1)
    x = torch.randn(3, 64, 384, device="cuda", dtype=torch.float32)
    y_r, _, gx_r, _, gw_r = _run(ref, x, None, False, fused_on=False, retain_res=False)
    y_f, _, gx_f, _, gw_f = _run(fused, x, None, False, fused_on=True, retain_res=False)
    assert torch.allclose(y_f, y_r, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gx_f, gx_r, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gw_f, gw_r, atol=1e-3, rtol=1e-3)


@needs_gpu
def test_model_forward_backward_matches_with_and_without_fusion():
    from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
    from mote.model.hnet import HNetForCausalLM

    cfg = MoteConfig(
        d_model_outer=64, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=2, d_model=64, n_heads=2, d_ff=128),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=128, n_candidates=3),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=512,
    )
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg, device="cuda")
    ids = torch.randint(0, 256, (2, 256), device="cuda")

    def pass_(fused_on):
        os.environ.pop("MOTE_NO_FUSED_NORM", None)
        if not fused_on:
            os.environ["MOTE_NO_FUSED_NORM"] = "1"
        try:
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids[:, :-1])
            # Next-byte cross-entropy, not logits.pow(2): the head masks the padding columns to -inf
            # (vocab_size -> pad_vocab_to, config.py), so squaring them gave inf, then nan gradients, then
            # a nan cosine similarity here. allclose on the logits still passed, because it counts equal
            # infinities as close — which is how this hid.
            loss = torch.nn.functional.cross_entropy(out.logits.float().flatten(0, 1), ids[:, 1:].reshape(-1))
            loss.backward()
            g = torch.cat([p.grad.flatten().float() for p in model.parameters() if p.grad is not None])
            return out.logits.detach().float(), out.routing.boundary_mask.detach(), g
        finally:
            os.environ.pop("MOTE_NO_FUSED_NORM", None)

    lg_ref, bm_ref, g_ref = pass_(False)
    lg_fus, bm_fus, g_fus = pass_(True)
    assert torch.equal(bm_ref, bm_fus)  # the router must draw the same chunks
    assert torch.allclose(lg_fus, lg_ref, atol=5e-2, rtol=5e-2)
    assert torch.nn.functional.cosine_similarity(g_ref, g_fus, dim=0) > 0.999
