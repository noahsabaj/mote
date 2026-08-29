"""Model-level checks: causality, cache/step equivalence, loss sanity, end-to-end decode equivalence."""

import math

import pytest
import torch
import torch.nn.functional as F

from mote.config import Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.dc import ratio_loss, atdc_target_ratio
from mote.model.hnet import HNetForCausalLM
from mote.model.mamba3 import HAS_MAMBA3_KERNEL, Mamba3Mixer
from mote.model.relation import FullRelation

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def small_cfg() -> MoteConfig:
    return MoteConfig(
        d_model_outer=64,
        encoder_layers=1,
        decoder_layers=1,
        main=RelationCfg(n_layers=2, d_model=96, n_heads=4, d_ff=128),
        mamba3=Mamba3Cfg(d_state=32, headdim=32, expand=2),
        max_seq_len=128,
    )


def test_relation_is_causal_and_cache_matches():
    torch.manual_seed(0)
    m = FullRelation(64, 4, layer_idx=1).to(DEV).float()
    x = torch.randn(2, 10, 64, device=DEV)
    y = m(x)
    x2 = x.clone()
    x2[:, 7:] += 1.0
    y2 = m(x2)
    assert torch.allclose(y[:, :7], y2[:, :7], atol=1e-5)
    y_pre, cache = m(x[:, :6], return_cache=True)
    y_rest = m(x[:, 6:], cache=cache)
    assert torch.allclose(torch.cat([y_pre, y_rest], 1), y, atol=1e-4)
    # exchange mass telemetry: first token has none, others in (0, 1)
    _, g = m(x, return_gates=True)
    assert torch.all(g[:, :, 0] == 0) and torch.all(g[:, :, 1:] > 0) and torch.all(g < 1)


def test_mamba3_step_matches_forward_and_state_continuation():
    torch.manual_seed(0)
    m = Mamba3Mixer(32, d_state=16, headdim=16, expand=2, layer_idx=0).to(DEV).float()
    u = torch.randn(2, 12, 32, device=DEV)
    y_full, st_full = m(u, return_final_states=True)
    st = m.allocate_inference_cache(2, DEV)
    ys = []
    for t in range(12):
        y_t, st = m.step(u[:, t : t + 1], st)
        ys.append(y_t)
    y_step = torch.cat(ys, 1)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / a.float().norm()).item()

    # the Triton kernel computes in bf16 internally (measured 4e-3 relative vs the fp32 reference);
    # the pure-PyTorch reference and the step port agree to ~1e-6.
    tol = 2e-2 if HAS_MAMBA3_KERNEL and DEV == "cuda" else 1e-4
    assert rel(y_full, y_step) < tol, rel(y_full, y_step)
    for a, b in zip(st_full, st):
        assert rel(a, b) < tol, rel(a, b)
    # prefix (kernel if available) + continuation with initial states (reference path) == full
    y_a, st_a = m(u[:, :5], return_final_states=True)
    y_b, st_b = m(u[:, 5:], return_final_states=True, initial_states=st_a)
    assert rel(torch.cat([y_a, y_b], 1), y_full) < tol
    for a, b in zip(st_full, st_b):
        assert rel(a, b) < tol
    # reference vs step must agree tightly regardless of kernels (initial_states now routes to the
    # kernel too — Input_States wiring — so invoke the reference path explicitly)
    zero = m.allocate_inference_cache(2, DEV)
    z, x, Bn, Cn, ADT, DT, trap, angles = m._preprocess(u)
    y_ref, _ = m._reference_forward(Cn, Bn, x, ADT, DT, trap, angles, z, zero)
    y_ref = m.out_proj(y_ref.reshape(u.shape[0], u.shape[1], -1).to(u.dtype))
    assert rel(y_ref, y_step) < 1e-4, rel(y_ref, y_step)


def test_chunked_ema_matches_sequential_recurrence():
    from mote.model.dc import DeChunkLayer

    torch.manual_seed(0)
    B, M, D = 2, 203, 16  # not a multiple of the 64-step block on purpose
    x = torch.randn(B, M, D, dtype=torch.float64)
    p = torch.rand(B, M, dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
    ref = torch.empty_like(x)
    z = torch.zeros(B, D, dtype=torch.float64)
    for t in range(M):
        z = p[:, t, None] * x[:, t] + (1 - p[:, t, None]) * z
        ref[:, t] = z
    out = DeChunkLayer._ema_chunked(x, p)
    assert (out - ref).abs().max().item() < 1e-9
    # gradients flow (no in-place hazards)
    xg = x.clone().requires_grad_(True)
    DeChunkLayer._ema_chunked(xg, p).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()


def test_ratio_loss_minimum_is_one_at_target():
    N = 6
    L = 600
    mask = torch.ones(1, L, dtype=torch.bool)
    bm = torch.zeros(1, L, dtype=torch.bool)
    bm[0, ::N] = True
    prob = torch.full((1, L, 2), 1.0 / N)
    val = ratio_loss(prob, bm, mask, N).item()
    assert abs(val - 1.0) < 1e-5
    assert atdc_target_ratio(0, 100, 5.0, 6.5, 0.6) == 5.0
    assert abs(atdc_target_ratio(100, 100, 5.0, 6.5, 0.6) - 6.5) < 1e-9


def test_hnet_forward_backward_and_decode_equivalence():
    torch.manual_seed(0)
    cfg = small_cfg()
    model = HNetForCausalLM(cfg).to(DEV).float()
    V = cfg.pad_vocab_to
    ids = torch.randint(0, 256, (2, 40), device=DEV)
    out = model(ids)
    assert out.logits.shape == (2, 40, V)
    assert torch.all(out.routing.boundary_mask[:, 0])
    loss = (
        F.cross_entropy(out.logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        + 0.03 * ratio_loss(out.routing.boundary_prob, out.routing.boundary_mask, torch.ones_like(ids, dtype=torch.bool), 6.0)
    )
    loss.backward()
    assert model.routing_module.q_proj_layer.weight.grad is not None  # STE + EMA carry gradient to the router
    assert model.main_network.layers[0].mixer.lam.grad is not None

    # batch-1 decode: prefill 25 bytes, step the remaining 15; must reproduce the full forward logits
    model.eval()
    with torch.no_grad():
        ids1 = ids[:1]
        full = model(ids1)
        st = model.allocate_inference_state(DEV)
        pre = model.prefill(ids1[:, :25], st)
        assert torch.allclose(pre.logits, full.logits[:, :25], atol=2e-3, rtol=2e-3), (pre.logits - full.logits[:, :25]).abs().max()
        assert torch.equal(pre.routing.boundary_mask, full.routing.boundary_mask[:, :25])
        for t in range(25, 40):
            lg, routing, is_b = model.step(ids1[:, t : t + 1], st)
            assert is_b == bool(full.routing.boundary_mask[0, t])
            assert torch.allclose(lg[:, 0], full.logits[:, t], atol=2e-3, rtol=2e-3), (t, (lg[:, 0] - full.logits[:, t]).abs().max())


def test_param_count_pilot_and_stage_groups():
    cfg = MoteConfig.mote_11m()
    model = HNetForCausalLM(cfg)
    n = model.num_params()
    assert 5e6 < n < 30e6, n
    groups = model.stage_param_groups()
    assert len(groups[0]) > 0 and len(groups[1]) > 0
    assert len(groups[0]) + len(groups[1]) == len(list(model.parameters()))
