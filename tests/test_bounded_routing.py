"""Bounded routing and ATDC's proficiency trigger (both added 2026-08-27).

The thing under test is a GUARANTEE the router never had. `target_ratio` looks like a compression
setting and is not one: `ratio_loss` has no gradient path to the achieved segmentation rate, so the
target is a pull the model may ignore — three trained Mote runs sat at val_bpic 3.20-3.32 while the
target ramped 5.0 -> 6.5, and 2605.30080's own ablation moved BPIC by 0.57 for 2.0 of target. The
projection (2602.17739 §3.3) is what actually bounds the count, so these tests are about the bound
holding and about it costing the fewest decisions when it binds.
"""

import math

import pytest
import torch

from mote.config import resolve_preset
from mote.model.dc import RoutingModule, atdc_target_ratio, project_boundaries
from mote.model.hnet import HNetForCausalLM


def _p(*vals):
    return torch.tensor([list(vals)], dtype=torch.float32)


def test_projection_is_a_noop_inside_the_bounds():
    p = _p(0.9, 0.1, 0.8, 0.2, 0.7, 0.3)
    b = p > 0.5
    m = torch.ones_like(b)
    out = project_boundaries(b, p, m, torch.tensor([1]), torch.tensor([5]))
    assert out.tolist() == b.tolist()


def test_ceiling_drops_the_least_confident_boundaries_and_nothing_else():
    p = _p(0.99, 0.95, 0.60, 0.55, 0.10, 0.20)
    b = p > 0.5  # four active
    m = torch.ones_like(b)
    out = project_boundaries(b, p, m, torch.tensor([0]), torch.tensor([2]))[0]
    assert out.sum() == 2
    assert out.tolist() == [True, True, False, False, False, False]  # the two highest-p survive


def test_floor_promotes_the_most_confident_non_boundaries_and_keeps_every_boundary():
    p = _p(0.99, 0.10, 0.45, 0.30, 0.05, 0.44)
    b = p > 0.5  # one active
    m = torch.ones_like(b)
    out = project_boundaries(b, p, m, torch.tensor([3]), torch.tensor([6]))[0]
    assert out.sum() == 3
    assert out[0] and out[2] and out[5]  # the original, then p=0.45 and p=0.44
    assert not out[1] and not out[3] and not out[4]


def test_projection_never_selects_a_padded_position():
    p = _p(0.9, 0.9, 0.9, 0.9)
    b = p > 0.5
    m = torch.tensor([[True, True, False, False]])
    out = project_boundaries(b, p, m, torch.tensor([4]), torch.tensor([4]))[0]
    assert out.tolist() == [True, True, False, False]  # the floor cannot be met from padding


def test_the_first_position_survives_any_ceiling():
    """Mote pads p[0] to 1.0, so the highest-confidence boundary is always the segment start — the
    paper's "the first position in each segment is always kept", for free."""
    p = _p(1.0, 0.95, 0.94, 0.93)
    b = p > 0.5
    m = torch.ones_like(b)
    out = project_boundaries(b, p, m, torch.tensor([0]), torch.tensor([1]))[0]
    assert out.tolist() == [True, False, False, False]


def test_each_row_of_a_batch_is_bounded_independently():
    p = torch.tensor([[0.9, 0.8, 0.7, 0.6], [0.9, 0.1, 0.1, 0.1]])
    b = p > 0.5
    m = torch.ones_like(b)
    out = project_boundaries(b, p, m, torch.tensor([2, 2]), torch.tensor([3, 3]))
    assert out[0].sum() == 3 and out[1].sum() == 2  # row 0 clipped down, row 1 topped up


def test_a_ceiling_caps_bytes_per_chunk_end_to_end():
    """What the whole thing is for: the boundary count a prompt produces becomes a number you can
    bound, which is what lets the serving arena be sized instead of grown."""
    cfg = resolve_preset("mote-32m")
    cfg.dc.target_ratio_init = 5.0
    cfg.dc.bound_ceiling = 1.2
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    ids = torch.randint(0, 256, (2, 1024))
    with torch.no_grad():
        out = model(ids)
    per_row = out.routing.boundary_mask.sum(-1)
    ceiling = math.ceil(1024 / 5.0 * 1.2)
    assert int(per_row.max()) <= ceiling
    assert float(1024 / int(per_row.max())) >= 1024 / ceiling  # i.e. bytes-per-chunk has a floor


def test_an_unconfigured_router_routes_exactly_as_before():
    cfg = resolve_preset("mote-32m")
    assert cfg.dc.bound_ceiling is None and cfg.dc.bound_floor == 0
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    ids = torch.randint(0, 256, (1, 512))
    with torch.no_grad():
        free = model(ids).routing.boundary_mask.clone()
    model.routing_module.bound_ceiling = 1e9  # a ceiling so loose it cannot bind
    with torch.no_grad():
        bounded = model(ids).routing.boundary_mask
    assert torch.equal(free, bounded)


def test_short_windows_are_not_bounded():
    """A short continuation is three bytes. Pricing a budget over it would force a boundary the model
    did not ask for, so the projection only applies once a window is long enough to have a rate."""
    r = RoutingModule(8)
    r.target_ratio, r.bound_ceiling, r.bound_floor = 5.0, 1.0, 0
    short = torch.ones(1, 3, dtype=torch.bool)
    p = torch.full((1, 3), 0.9)
    assert torch.equal(r._bound(short, p, torch.ones_like(short)), short)
    long_mask = torch.ones(1, r.bound_min_len, dtype=torch.bool)
    p_long = torch.full((1, r.bound_min_len), 0.9)
    assert int(r._bound(long_mask, p_long, torch.ones_like(long_mask)).sum()) < r.bound_min_len


def test_decode_uses_its_own_threshold_not_the_projection():
    """One byte has no sequence to project over (signed 2026-08-27: bound everywhere, decode via a
    calibrated fixed threshold)."""
    cfg = resolve_preset("mote-1m")
    cfg.dc.decode_threshold = 0.99
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    state = model.allocate_inference_state("cpu")
    with torch.no_grad():
        model.prefill(torch.randint(0, 256, (1, 32)), state)
        fired = 0
        for _ in range(32):
            _, routing, is_b = model.step(torch.tensor([[65]]), state)
            fired += int(is_b)
            assert bool(is_b) == bool(routing.boundary_prob[0, 1] > 0.99)
    model.routing_module.decode_threshold = 0.0
    state2 = model.allocate_inference_state("cpu")
    with torch.no_grad():
        model.prefill(torch.randint(0, 256, (1, 32)), state2)
        always = sum(int(model.step(torch.tensor([[65]]), state2)[2]) for _ in range(32))
    assert always == 32 and always > fired


# --- ATDC's proficiency trigger --------------------------------------------------------------
def test_the_schedule_alone_is_unchanged_when_no_threshold_is_set():
    for step in (0, 300, 600, 800, 1000):
        assert atdc_target_ratio(step, 1000, 5.0, 6.5, 0.6) == pytest.approx(
            atdc_target_ratio(step, 1000, 5.0, 6.5, 0.6, loss_window=0.01, threshold=None))


def test_the_trigger_boosts_only_while_the_loss_is_under_tau():
    base = atdc_target_ratio(800, 1000, 5.0, 6.5, 0.6)
    hot = atdc_target_ratio(800, 1000, 5.0, 6.5, 0.6, loss_window=0.9, threshold=0.5, rate=1.05)
    cool = atdc_target_ratio(800, 1000, 5.0, 6.5, 0.6, loss_window=0.4, threshold=0.5, rate=1.05)
    assert hot == pytest.approx(base)
    assert cool == pytest.approx(base * 1.05)


def test_the_trigger_waits_for_a_full_window():
    """`loss_window=None` is how the trainer says "not enough steps yet" (Alg. 1 lines 13-14)."""
    base = atdc_target_ratio(800, 1000, 5.0, 6.5, 0.6)
    assert atdc_target_ratio(800, 1000, 5.0, 6.5, 0.6, loss_window=None, threshold=0.5) == pytest.approx(base)


# --- the floor is a rate (2026-08-29) ---------------------------------------------------------------
# `--bound-floor 2048` means 2048 boundaries per 16384-byte window (≤ 8 bytes a chunk), not 2048 boundaries in
# whatever window the router is handed. Read as an absolute count it never bound in training (the router
# routes ~4.4k boundaries at 16384) and, at serving, forced every byte of a continuation shorter than 2048
# bytes to be a boundary and 2 bytes a chunk on a 4096-byte prefill window.

def _router(floor: int, window: int) -> RoutingModule:
    r = RoutingModule(8)
    r.bound_floor, r.bound_window, r.target_ratio = floor, window, 5.0
    return r


def _mask_for(r: RoutingModule, L: int, p_high: int):
    """A window of L positions whose first `p_high` positions are confident boundaries."""
    p = torch.zeros(1, L)
    p[0, :p_high] = 0.9
    p[0, p_high:] = 0.1
    b = p > 0.5
    return r._bound(b, p, torch.ones_like(b)), b


def test_floor_scales_with_the_window():
    r = _router(floor=2048, window=16384)
    out, b = _mask_for(r, L=16384, p_high=4400)
    assert out.sum() == 4400  # the trunk's natural rate is above the floor: untouched (bitwise the old behaviour)
    out, b = _mask_for(r, L=4096, p_high=1200)
    assert out.sum() == 1200  # a 4096-byte prefill window: floor = 512, natural 1200 -> untouched
    out, b = _mask_for(r, L=100, p_high=28)
    assert out.sum() == 28  # a 100-byte continuation: floor = ceil(100 * 2048 / 16384) = 13 -> untouched


def test_floor_still_binds_when_the_router_collapses():
    r = _router(floor=2048, window=16384)
    out, _ = _mask_for(r, L=4096, p_high=100)  # 41 bytes a chunk: the guardrail fires at 512
    assert out.sum() == 512


def test_absolute_floor_is_kept_for_old_configs():
    r = _router(floor=2048, window=0)
    out, _ = _mask_for(r, L=100, p_high=28)
    assert out.sum() == 100  # the old reading: every byte of a short window (what this fix removes from new configs)


def test_model_wires_the_training_window_into_the_floor():
    cfg = resolve_preset("mote-1m")
    cfg.dc.bound_floor, cfg.max_seq_len = 2048, 16384
    m = HNetForCausalLM(cfg, device="cpu").eval()
    assert m.routing_module.bound_window == 16384
    st = m.allocate_inference_state("cpu")
    m.prefill(torch.randint(0, 256, (1, 256)), st)
    _, bm, _ = m.forward_from_state(torch.randint(0, 256, (1, 100)), st)
    assert int(bm.sum()) < 100  # no longer every byte
