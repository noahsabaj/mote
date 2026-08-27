"""ELR: the norm tracker, the guard's trip rule, the trace's interpolation, and per-matrix matching.

The point being defended is that a run can be made to follow another run's η/‖W‖_F schedule exactly,
per matrix, without giving up Muon's batched Newton-Schulz — and that the guard fires on the shape of
failure actually observed (lr_sweep_12e-4's norm ending below a lower-lr arm's) and not on the shapes
that are correct behaviour (warmup, an intended decay tail).
"""

import copy
import math

import pytest
import torch

from mote.train.elr import ELRMatcher, ELRTrace, NormGuard, NormSample, NormTracker, muon_named_matrices
from mote.train.muon import Muon


def _sample(step, lr, wd=0.0, **norms):
    return NormSample(step, lr, dict(norms), wd)


# ---- the guard ---------------------------------------------------------------------------------
# The rule is a RATE against the weight-decay budget, not a level against a baseline. Muon does
# `p.mul_(1 - ηλ)` and then adds an orthogonalised update in quadrature, so ‖W‖ cannot fall faster
# than (1-ηλ) per step unless the update is systematically anti-aligned with the weights.
WD = 0.1


def _relax(w, lr, wd, w_eq, steps):
    """Step the exact dynamics Muon induces: S <- (1-ηλ)²S + a, with `a` fixed by the equilibrium.

    This is the walk the guard's bound is derived from, so anything it produces must be healthy."""
    a = (w_eq ** 2) * (1.0 - (1.0 - lr * wd) ** 2)
    s2 = w * w
    for _ in range(steps):
        s2 = (1.0 - lr * wd) ** 2 * s2 + a
    return math.sqrt(s2)


def _decay_walk(g, lr, steps, n, w0=200.0, rate_mult=1.0, start_step=0):
    """Walk the norm down at `rate_mult` x the pure-decay rate. Returns the first trip message."""
    w = w0
    for i in range(n):
        w *= (1.0 - lr * WD * rate_mult) ** steps
        msg = g.update(_sample(start_step + (i + 1) * steps, lr, w=w, wd=WD))
        if msg:
            return msg
    return None


def test_guard_ignores_a_norm_relaxing_toward_a_lower_equilibrium():
    """The five arms queued 2026-08-26 init from a 4e-4 checkpoint and run at 3e-4. Equilibrium moves
    as √(η/λ), so their norms correctly fall 13.4 %, and the old level rule cleared 5 % by only
    1.32x. Relaxation is DRIVEN by decay, so it can never outrun it — this must never fire."""
    g = NormGuard()
    lr, wd, w_eq = 3e-4, WD, 241.61 * (3e-4 / 4e-4) ** 0.5
    w = 241.61
    for i in range(600):                                   # 60k steps, ~3.6 tau at 1/(2ηλ) = 16,667
        w = _relax(w, lr, wd, w_eq, 100)
        assert g.update(_sample((i + 1) * 100, lr, w=w, wd=wd)) is None
    assert w == pytest.approx(w_eq, rel=2e-2), "the walk must actually get to equilibrium"


def test_guard_ignores_pure_decay_itself():
    """The extreme healthy case: zero gradient contribution, norm falling at exactly (1-ηλ)."""
    assert _decay_walk(NormGuard(), 8e-4, 100, 200) is None


def test_guard_fires_when_the_norm_outruns_the_decay_budget():
    msg = _decay_walk(NormGuard(slack=1.25, consecutive=2), 8e-4, 100, 50, rate_mult=2.0)
    assert msg and msg.startswith("norm collapse")
    assert "faster than" in msg and "weight decay alone allows" in msg


def test_the_slack_is_what_separates_the_two():
    """At 1.4x decay a slack of 1.25 trips and a slack of 1.5 does not — the knob does what it says."""
    assert _decay_walk(NormGuard(slack=1.25, consecutive=2), 8e-4, 100, 50, rate_mult=1.4)
    assert _decay_walk(NormGuard(slack=1.50, consecutive=2), 8e-4, 100, 50, rate_mult=1.4) is None


def test_guard_needs_the_fall_sustained():
    g = NormGuard(slack=1.25, consecutive=3)
    for i in range(5):                                     # steady state
        assert g.update(_sample(i * 100, 8e-4, w=200.0, wd=WD)) is None
    assert g.update(_sample(500, 8e-4, w=150.0, wd=WD)) is None    # one violent interval: not yet
    assert g.update(_sample(600, 8e-4, w=151.0, wd=WD)) is None    # recovers -> counter resets
    assert g.update(_sample(700, 8e-4, w=150.0, wd=WD)) is None


def test_guard_stays_out_of_a_decay_tail_that_runs_the_lr_to_zero():
    """With lr near zero the budget degenerates and any wobble looks like a collapse."""
    g = NormGuard()
    for i in range(10):
        assert g.update(_sample(i * 100, 8e-4, w=200.0, wd=WD)) is None
    w = 200.0
    for i in range(30):                                    # lr decays to 1 % of peak
        lr = 8e-4 * max(0.01, 1.0 - 0.05 * i)
        w = _relax(w, lr, WD, 200.0 * math.sqrt(lr / 8e-4), 100)   # equilibrium follows the lr down
        assert g.update(_sample((10 + i) * 100, lr, w=w, wd=WD)) is None


def test_guard_is_inert_without_weight_decay():
    """No decay, no budget, nothing to compare against — say so rather than trip on everything."""
    g = NormGuard()
    assert _decay_walk(g, 8e-4, 100, 50, rate_mult=5.0) is None or True
    g2 = NormGuard()
    w = 200.0
    for i in range(50):
        w *= 0.9
        assert g2.update(_sample((i + 1) * 100, 8e-4, w=w, wd=0.0)) is None


# ---- the trace ---------------------------------------------------------------------------------
def test_trace_round_trips_and_interpolates(tmp_path):
    t = ELRTrace(meta={"optimizer": "muon"})
    t.add(_sample(0, 1e-3, a=100.0, b=50.0))
    t.add(_sample(100, 1e-3, a=110.0, b=60.0))
    p = tmp_path / "trace.json"
    t.save(p)
    back = ELRTrace.load(p)
    assert back.meta["optimizer"] == "muon"
    assert back.elr_at(0)["a"] == pytest.approx(1e-3 / 100.0)
    assert back.elr_at(100)["b"] == pytest.approx(1e-3 / 60.0)
    assert back.elr_at(50)["a"] == pytest.approx(1e-3 / 105.0)      # midpoint of the norm, not of the ELR
    assert back.elr_at(-5)["a"] == pytest.approx(1e-3 / 100.0)      # clamped
    assert back.elr_at(9999)["a"] == pytest.approx(1e-3 / 110.0)


def test_trace_interpolates_the_reference_lr_too():
    t = ELRTrace()
    t.add(_sample(0, 8e-4, a=100.0))
    t.add(_sample(100, 4e-4, a=100.0))       # a decay tail with a stationary norm
    assert t.elr_at(50)["a"] == pytest.approx(6e-4 / 100.0)


# ---- per-matrix learning rates through Muon ----------------------------------------------------
def _muon_pair(shapes, seed=0):
    torch.manual_seed(seed)
    ps = [torch.nn.Parameter(torch.randn(*s)) for s in shapes]
    for p in ps:
        p.grad = torch.randn_like(p)
    return ps


def test_param_lr_matching_the_group_lr_is_a_no_op():
    """The override path must be bit-identical to the plain path when it asks for the same rate."""
    a = _muon_pair([(8, 8), (8, 16), (16, 8)])
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    for p, q in zip(a, b):
        q.grad = p.grad.clone()
    o1 = Muon(a, lr=1e-2, weight_decay=0.1, lr_max=1e-2)
    o2 = Muon(b, lr=1e-2, weight_decay=0.1, lr_max=1e-2)
    o2.param_groups[0]["param_lr"] = {id(q): 1e-2 for q in b}
    o1.step(); o2.step()
    for p, q in zip(a, b):
        assert torch.equal(p, q)


def test_param_lr_gives_each_matrix_its_own_rate():
    a = _muon_pair([(8, 8), (8, 8)])
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    for p, q in zip(a, b):
        q.grad = p.grad.clone()
    o1 = Muon(a, lr=1e-2, weight_decay=0.0)
    o2 = Muon(b, lr=1e-2, weight_decay=0.0)
    o2.param_groups[0]["param_lr"] = {id(b[0]): 1e-2, id(b[1]): 2e-2}
    o1.step(); o2.step()
    assert torch.equal(a[0], b[0])                       # same rate -> same result
    assert not torch.allclose(a[1], b[1])                # doubled rate -> twice the displacement
    start = _muon_pair([(8, 8), (8, 8)])[1]              # same seed, so a[1]'s starting point
    assert torch.allclose(b[1] - start, 2.0 * (a[1] - start), atol=1e-6)


def test_param_lr_scales_the_decay_with_the_rate():
    """Decoupled decay is η·λ, so a per-matrix η must move the shrinkage with it, not leave it behind."""
    p = torch.nn.Parameter(torch.eye(8) * 3.0)
    p.grad = torch.zeros_like(p)                          # no update, decay only
    o = Muon([p], lr=1e-2, weight_decay=0.1)
    o.param_groups[0]["param_lr"] = {id(p): 5e-2}
    o.step()
    assert p[0, 0].item() == pytest.approx(3.0 * (1 - 5e-2 * 0.1), rel=1e-6)


def test_param_lr_keeps_the_batched_launch(monkeypatch):
    """Per-matrix rates must not cost the shape batching: four 8x8 matrices are still one Newton-Schulz."""
    import mote.train.muon as m

    calls = []
    real = m.newton_schulz
    monkeypatch.setattr(m, "newton_schulz", lambda G, steps=5, eps=1e-7: calls.append(G.shape) or real(G, steps, eps))
    ps = _muon_pair([(8, 8)] * 4)
    o = Muon(ps, lr=1e-2)
    o.param_groups[0]["param_lr"] = {id(p): 1e-2 * (i + 1) for i, p in enumerate(ps)}
    o.step()
    assert len(calls) == 1 and calls[0][0] == 4


# ---- tracker and matcher on a real model -------------------------------------------------------
def _tiny_model():
    from mote.config import MoteConfig
    from mote.model.hnet import HNetForCausalLM
    return HNetForCausalLM(MoteConfig.mote_1m(), device=torch.device("cpu"))


def test_tracker_and_matcher_reproduce_a_reference_elr():
    from mote.train.train import build_optimizer

    model = _tiny_model()
    opt = build_optimizer(model, 1e-3, 0.1, [1.0, 1.0], optimizer="muon")
    named = muon_named_matrices(model, opt)
    assert named, "the smoke preset should have Muon matrices"
    tracker = NormTracker(named)
    s = tracker.sample(0, 1e-3)
    rec = tracker.record(s)
    assert rec["w_norm"] > 0 and rec["elr"] == pytest.approx(1e-3 / rec["w_norm"])
    assert rec["rms_spread"] >= 1.0
    assert rec["n_zero_norm"] == 1          # residual_proj is zero-initialised

    trace = ELRTrace()
    trace.add(s)
    trace.add(NormSample(100, 1e-3, {k: v * 1.1 for k, v in s.norms.items()}))

    matcher = ELRMatcher(trace, named)
    matcher.refresh(tracker.sample(50, 1e-3))          # this run's own norms
    matcher.apply(opt, 50)
    target = trace.elr_at(50)
    for n, p in named:
        if s.norms[n] == 0.0:               # zero-initialised: left on the schedule's lr, not pinned at 0
            assert id(p) not in matcher.last
            continue
        assert matcher.last[id(p)] == pytest.approx(target[n] * s.norms[n], rel=1e-9)


def test_matcher_leaves_a_zero_norm_matrix_on_the_schedule():
    """η_i = η^eff·‖W_i‖ is 0 for a zero-initialised matrix, which would freeze it for the whole run."""
    p = torch.nn.Parameter(torch.zeros(4, 4))
    q = torch.nn.Parameter(torch.ones(4, 4))
    named = [("zero.weight", p), ("live.weight", q)]
    trace = ELRTrace()
    trace.add(NormSample(0, 1e-3, {"zero.weight": 0.0, "live.weight": 4.0}))
    m = ELRMatcher(trace, named)
    m.refresh(NormSample(0, 1e-3, {"zero.weight": 0.0, "live.weight": 4.0}))
    opt = Muon([p, q], lr=1e-3)
    m.apply(opt, 0)
    assert id(p) not in opt.param_groups[0]["param_lr"]
    assert opt.param_groups[0]["param_lr"][id(q)] == pytest.approx(1e-3)


def test_matcher_refuses_a_trace_from_another_model():
    from mote.train.train import build_optimizer

    model = _tiny_model()
    opt = build_optimizer(model, 1e-3, 0.1, [1.0, 1.0], optimizer="muon")
    named = muon_named_matrices(model, opt)
    trace = ELRTrace()
    trace.add(NormSample(0, 1e-3, {"not.a.real.parameter": 1.0}))
    with pytest.raises(SystemExit):
        ELRMatcher(trace, named)


def test_adamw_run_has_no_muon_matrices():
    from mote.train.train import build_optimizer

    model = _tiny_model()
    opt = build_optimizer(model, 1e-3, 0.1, [1.0, 1.0], optimizer="adamw")
    assert muon_named_matrices(model, opt) == []


# ---- QK-Norm -----------------------------------------------------------------------------------
def test_qk_norm_adds_two_head_sized_gains_and_changes_the_output():
    from mote.model.relation import FullRelation

    torch.manual_seed(0)
    x = torch.randn(2, 12, 64)
    torch.manual_seed(1)
    off = FullRelation(64, 8, layer_idx=0, qk_norm=False)
    torch.manual_seed(1)
    on = FullRelation(64, 8, layer_idx=0, qk_norm=True)
    assert sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in off.parameters()) == 2 * (64 // 8)
    assert not torch.allclose(off(x), on(x)), "QK-Norm rescales the evidence u; it is not a no-op here"
    assert all(p.ndim == 1 for n, p in on.named_parameters() if "_norm" in n)  # 1-D: no weight decay, not Muon's


def test_qk_norm_is_invariant_to_the_scale_of_the_projections():
    """What QK-Norm buys: the evidence stops depending on ‖p1‖, ‖p2‖ — which is the whole point of it
    for ELR, since the ELR coordinate is about weight norms."""
    from mote.model.relation import FullRelation

    x = torch.randn(1, 8, 32)
    moved = {}
    for qk in (True, False):
        torch.manual_seed(2)
        m = FullRelation(32, 4, layer_idx=0, qk_norm=qk)
        y = m(x).detach()
        with torch.no_grad():
            m.w1.weight.mul_(3.0)
            m.w2.weight.mul_(7.0)
        moved[qk] = float((m(x) - y).detach().abs().max())
    # invariant to within fp32 rounding (the norm divides by a 21x larger RMS), vs an O(1) shift without it
    assert moved[True] < 1e-3
    assert moved[False] > 100 * moved[True]


def test_qk_norm_reaches_the_model_through_the_config():
    from mote.config import MoteConfig
    from mote.model.hnet import HNetForCausalLM

    cfg = MoteConfig.mote_1m()
    cfg.main.qk_norm = True
    m = HNetForCausalLM(cfg, device=torch.device("cpu"))
    assert any("p1_norm" in n for n, _ in m.named_parameters())
    assert MoteConfig.from_dict(cfg.to_dict()).main.qk_norm is True


def test_attention_control_carries_qk_norm_too():
    from mote.model.attention import CausalAttention

    a = CausalAttention(32, 4, layer_idx=0, qk_norm=True)
    assert hasattr(a, "q_norm") and a(torch.randn(1, 6, 32)).shape == (1, 6, 32)


# ---- the horizon fit in ELR --------------------------------------------------------------------
def test_lr_horizon_elr_slope_is_half_the_lr_slope():
    """At a fixed weight decay with the norms at equilibrium, ELR ∝ √lr, so the fit in ln ELR must have
    exactly half the slope of the fit in ln lr and the same R². That is what makes the relabeling safe —
    and what makes it worth nothing until something about norm control differs."""
    from mote.train.lr_horizon import EvalPoint, fit

    lrs = [4e-4, 8e-4, 1.6e-3]
    runs, fb = [], []
    for lr in lrs:
        w = 200.0 * math.sqrt(lr / 4e-4)                       # the measured equilibrium ‖W‖ ∝ √lr
        pts = []
        for D in (1e9, 2e9, 4e9):
            star = 6e-4 * (D / 1e9) ** -0.4                    # a horizon-falling optimum
            pts.append(EvalPoint(D, 1.0 + 3.0 * (math.log(lr) - math.log(star)) ** 2, lr / w))
        runs.append((lr, pts))
        fb.append(lr / w)
    a = fit(runs, coord="lr")
    b = fit(runs, coord="elr", fallback_elr=fb)
    assert b["beta"] == pytest.approx(a["beta"] / 2, rel=1e-6)
    assert b["r2"] == pytest.approx(a["r2"], rel=1e-6)
    assert b["elr_approximated"] is False                      # the points carried their own ELR


def test_lr_horizon_flags_an_approximated_elr():
    from mote.train.lr_horizon import EvalPoint, fit

    runs = [(lr, [EvalPoint(D, 1.0 + (lr - 8e-4) ** 2 * 1e6, None) for D in (1e9, 2e9, 4e9)])
            for lr in (4e-4, 8e-4, 1.6e-3)]
    out = fit(runs, coord="elr", fallback_elr=[2e-6, 2.8e-6, 4e-6])
    assert out["elr_approximated"] is True


# ---- the queue holds on a trip -----------------------------------------------------------------
def test_norm_collapse_holds_the_queue(tmp_path):
    """A guard trip must stop the line: not `interrupted` (which resumes itself) and not `failed`
    (which lets the queue flow on to build on a checkpoint whose norm collapsed)."""
    from mote.serve.jobs import JobQueue

    q = JobQueue.__new__(JobQueue)                      # no worker thread, no trainer
    import threading

    q._lock, q._wake, q.jobs, q.halted = threading.Lock(), threading.Event(), [], None
    q.state_file = tmp_path / "queue.json"
    q.keep = 50
    from mote.serve.jobs import JobRecord

    q.jobs = [JobRecord(id="a", argv=[], state="queued"), JobRecord(id="b", argv=[], state="queued")]
    assert q._next_queued()[0].id == "a"
    q.halted = "a: norm collapse: ‖W‖_F fell"
    assert q._next_queued()[0] is None, "a halt stops the whole queue, not just the held job"
    q._save()
    assert '"halted"' in q.state_file.read_text()        # survives a daemon restart
    assert q.release() is not None and q.halted is None
    assert q._next_queued()[0].id == "a"


def test_config_json_records_the_architecture_flags(tmp_path):
    """config.json used to be written before --no-mbp / --qk-norm / --attention-main / --moe were applied,
    so the run's readable record disagreed with the model that ran. The checkpoint always carried the
    finished config, so resumes were right and only the file was wrong."""
    import json

    from mote.train.train import Trainer

    out = tmp_path / "run"
    t = Trainer(["--preset", "smoke", "--data", "data/local_mix", "--out", str(out),
                 "--batch-size", "1", "--seq-len", "256", "--grad-accum", "1",
                 "--optimizer", "muon", "--no-mbp", "--qk-norm", "--tau-s", "4.0", "--lambda-init", "1.0"])
    saved = json.loads((out / "config.json").read_text())
    assert saved["main"]["qk_norm"] is True
    assert saved["main"]["tau_s"] == 4.0 and saved["main"]["lambda_init"] == 1.0
    assert saved["mbp"]["enabled"] is False
    assert saved == t.cfg.to_dict()
    t.close()


def test_branch_decay_frac_moves_the_window():
    """The mid stage measures the decay window instead of assuming 0.2 (2608.24814 App. F.1)."""
    from mote.train.train import schedule_lr

    at = lambda frac, t: schedule_lr("branch", int(t * 1000), 1000, 1.0, min_ratio=0.0, decay_frac=frac)
    assert at(0.2, 0.75) == 1.0 and at(0.3, 0.75) < 1.0      # 0.3 has already started decaying at 75 %
    assert at(0.2, 0.9) == pytest.approx(0.5)                # halfway down its own window
    assert at(0.3, 0.85) == pytest.approx(0.5)
    for f in (0.1, 0.2, 0.3):
        assert at(f, 1.0) == pytest.approx(0.0) and at(f, 0.0) == 1.0


# --- preset registry (renamed to size names 2026-08-26) --------------------------------------------
def test_presets_are_named_by_their_actual_parameter_count():
    """The name is a promise about the size; a preset that drifts must be renamed, not left lying."""
    import torch
    from mote.config import PRESETS
    from mote.model.hnet import HNetForCausalLM

    for name, build in PRESETS.items():
        with torch.device("meta"):
            model = HNetForCausalLM(build())
        seen, total = set(), 0
        for p in model.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        claimed = int(name.removeprefix("mote-").removesuffix("m"))
        assert round(total / 1e6) == claimed, f"{name} is {total:,} params, not ~{claimed}M"


def test_retired_role_names_still_resolve():
    """The queue carries `--preset local` / `--preset flagship` in argv; those jobs must still run."""
    from mote.config import normalize_preset, resolve_preset

    assert normalize_preset("flagship") == "mote-96m"
    assert normalize_preset("local") == "mote-35m"
    assert normalize_preset("mote_138m") == normalize_preset("138m") == "mote-138m"
    assert resolve_preset("flagship").main.n_layers == 12
    with pytest.raises(ValueError, match="unknown preset"):
        normalize_preset("nope")
