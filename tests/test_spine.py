"""The hyper-connection spine (mote/model/spine.py), signed 2026-08-26.

Several of these are characterisation tests rather than correctness tests: they pin the properties
each manifold is *claimed* to have in its paper, so a control arm that quietly stops being the thing
it is controlling for fails here instead of in a run.
"""

import math

import pytest
import torch

from mote.model.spine import (
    LSS_STD,
    S_BIAS,
    S_BIAS_FRAC,
    ResMixer,
    Spine,
    StreamExpand,
    _cayley,
    _skew,
    helmert,
    n_cols,
)

PROJECTIONS = ("spectral_sphere", "orthogonal", "sinkhorn", "perm_convex", "diag", "none")
AFFINE = ("spectral_sphere", "orthogonal", "sinkhorn", "perm_convex")  # rows/cols sum to 1


@pytest.mark.parametrize("n", [2, 3, 4, 8])
def test_helmert_is_an_orthonormal_basis_of_the_zero_mean_subspace(n):
    """P Pᵀ = I − J is the whole reason `M = I` gives `H_res = J + (I−J) = I`."""
    p = helmert(n)
    assert p.shape == (n, n - 1)
    assert torch.allclose(p.T @ p, torch.eye(n - 1), atol=1e-6)          # orthonormal columns
    assert torch.allclose(p @ p.T, torch.eye(n) - 1.0 / n, atol=1e-6)    # projector onto 1⊥
    assert torch.allclose(p.T @ torch.ones(n), torch.zeros(n - 1), atol=1e-6)


def test_cayley_of_a_skew_matrix_is_orthogonal_and_is_the_identity_at_zero():
    torch.manual_seed(0)
    for m in (1, 3, 7):
        k = m * (m - 1) // 2
        a = _skew(torch.randn(5, k) * 2.0, m)
        assert torch.allclose(a, -a.transpose(-1, -2), atol=1e-6)
        q = _cayley(a)
        eye = torch.eye(m).expand_as(q)
        assert torch.allclose(q @ q.transpose(-1, -2), eye, atol=1e-5)
        assert torch.allclose(_cayley(_skew(torch.zeros(k), m)), torch.eye(m), atol=1e-6)


@pytest.mark.parametrize("project", PROJECTIONS)
def test_every_projection_is_the_identity_at_its_own_initialization(project):
    """Every variant must start as a plain residual connection, or the arms are not comparable."""
    mixer = ResMixer(4, project)
    h = mixer(mixer.bias_init().expand(3, 5, -1).clone())
    # Each family misses the identity by a different, derivable amount — none of them can hit it
    # exactly, because every bound they impose is enforced by a saturating function.
    tol = {
        "spectral_sphere": 1 - math.tanh(S_BIAS),           # Σ = tanh(3) = 0.99505
        "orthogonal": 1e-6,                                 # Cayley(0) is exactly I
        "sinkhorn": 1e-3,                                   # exp(-8) off-diagonal, then normalised
        "perm_convex": 24 * math.exp(-8),                   # softmax mass leaking to the other n! − 1
        "diag": 1 - 1 / (1 + math.exp(-6)),                 # σ(6) = 0.99753
        "none": 1e-6,                                       # the bias is literally the identity
    }[project]
    assert torch.allclose(h, torch.eye(4).expand_as(h), atol=max(tol, 1e-6)), h[0, 0]


@pytest.mark.parametrize("project", [p for p in PROJECTIONS if p != "none"])
def test_constrained_projections_are_non_expansive_far_from_initialization(project):
    """‖H_res‖₂ ≤ 1 is the whole stability argument; it must hold for adversarially large inputs."""
    torch.manual_seed(0)
    h = ResMixer(4, project)(torch.randn(2048, n_cols(project, 4)) * 5.0)
    assert torch.linalg.matrix_norm(h, 2).max() <= 1.02


def test_unconstrained_hyper_connections_do_explode():
    """The control arm has to actually be uncontrolled, or it controls for nothing (2512.24880 §3.1
    measured a composite gain of 3000 at 27B; this is the same failure in miniature)."""
    torch.manual_seed(0)
    h = ResMixer(4, "none")(torch.randn(2048, 16) * 5.0)
    assert torch.linalg.matrix_norm(h, 2).max() > 5.0


@pytest.mark.parametrize("project", AFFINE)
def test_affine_projections_conserve_the_mean_across_streams(project):
    torch.manual_seed(0)
    h = ResMixer(4, project)(torch.randn(512, n_cols(project, 4)) * 3.0)
    ones = torch.ones(4)
    assert torch.allclose(h @ ones, ones.expand(512, 4), atol=1e-4)          # rows sum to 1
    if project != "sinkhorn":  # see the characterisation test below
        assert torch.allclose(h.transpose(-1, -2) @ ones, ones.expand(512, 4), atol=1e-4)


def test_the_spectral_sphere_admits_the_negative_entries_birkhoff_forbids():
    """The point of sHC over mHC: subtractive mixing. Non-negativity is what mHC pays for its norm
    bound, and it is why its matrices decay toward identity (2603.20896 §4.1)."""
    torch.manual_seed(0)
    raw = torch.randn(1024, n_cols("spectral_sphere", 4)) * 3.0
    assert ResMixer(4, "spectral_sphere")(raw).min() < -0.1
    assert ResMixer(4, "sinkhorn")(torch.randn(1024, 16) * 3.0).min() >= 0.0
    assert ResMixer(4, "perm_convex")(torch.randn(1024, 24) * 3.0).min() >= 0.0


def test_the_birkhoff_polytope_sits_inside_the_spectral_sphere():
    """`B_n ⊂ S_n`, so choosing the sphere cannot cost expressivity relative to mHC."""
    torch.manual_seed(0)
    doubly_stochastic = ResMixer(4, "perm_convex")(torch.randn(512, 24) * 3.0)
    assert torch.linalg.matrix_norm(doubly_stochastic, 2).max() <= 1.0 + 1e-4


def test_twenty_sinkhorn_iterations_do_not_actually_converge():
    """Characterises mHC's known defect (2601.05732 §4.3) so the control arm stays honest: a finite
    Sinkhorn-Knopp run leaves the columns off 1 and ‖H‖₂ slightly above the bound it claims."""
    torch.manual_seed(0)
    h = ResMixer(4, "sinkhorn")(torch.randn(4096, 16) * 3.0)
    col_error = (h.transpose(-1, -2) @ torch.ones(4) - 1.0).abs().max()
    assert col_error > 1e-3, "if this ever passes exactly, mHC's projection changed"
    assert torch.linalg.matrix_norm(h, 2).max() > 1.0


# --- the spine itself -----------------------------------------------------------------------------
def _run(mode, n, sites=7, d=64, lss=False, y_scale=0.0, seed=0):
    torch.manual_seed(seed)
    h = torch.randn(2, 32, d)
    expand = StreamExpand(d, n, mode, lss=lss)
    spine = [Spine(d, n, site_idx=i, mode=mode) for i in range(sites)]
    x = expand(h)
    for s in spine:
        u, carried = s.read(x)
        x = s.write(x, u * y_scale, carried)
    return h, expand.collapse(x), expand, spine


@pytest.mark.parametrize("n", [2, 4])
def test_expand_is_exactly_the_identity_at_init_when_the_streams_start_equal(n):
    """Identical streams pass a mean-conserving mixer untouched — (I−J)X is zero, so whatever the
    singular values are doing cannot attenuate the residual. That is the affine constraint earning
    its place, and it is why `expand` needs no saturation headroom that `frac` does."""
    h, out, _, _ = _run("expand", n, lss=False)
    assert torch.allclose(out, h, atol=1e-6)


@pytest.mark.parametrize("n", [2, 4])
def test_frac_starts_within_half_a_percent_of_the_identity(n):
    """Slices differ, so (I−J)X ≠ 0 and tanh(S_BIAS_FRAC) attenuates once per site."""
    h, out, _, _ = _run("frac", n, lss=False)
    err = ((out - h).norm() / h.norm()).item()
    assert err < 0.01, err
    assert math.isclose(err, 1 - math.tanh(S_BIAS_FRAC) ** 7, abs_tol=0.01)


def test_learned_stream_scaling_breaks_the_symmetry_it_is_there_to_break():
    h, out, expand, _ = _run("expand", 4, lss=True)
    assert not torch.allclose(out, h, atol=1e-4)                    # streams are no longer copies
    assert ((out - h).norm() / h.norm()).item() < 0.05              # but only just
    assert expand.scale.std().item() == pytest.approx(LSS_STD, rel=0.5)


def test_a_uniform_read_out_would_kill_the_last_mixer():
    """A mean cannot see a mean-conserving mixer: dL/dH_res is constant down the destination axis,
    H_res's columns sum to 1, and the two contract to zero. `StreamExpand.collapse` therefore uses a
    learned convex read; this test is what stops someone simplifying it back to `.mean(-2)`."""
    _, out, expand, spine = _run("expand", 4, lss=True, y_scale=0.1)
    out.square().mean().backward()
    assert all(s.b_res.grad.abs().max() > 1e-6 for s in spine)

    _, out2, expand2, spine2 = _run("expand", 4, lss=True, y_scale=0.1, seed=1)
    with torch.no_grad():
        expand2.read_out.zero_()                                     # softmax(0) == a plain mean
    x = expand2(torch.randn(2, 32, 64))
    for s in spine2:
        u, carried = s.read(x)
        x = s.write(x, u * 0.1, carried)
    torch.einsum("...nd,n->...d", x, expand2.read_out.softmax(-1)).square().mean().backward()
    assert spine2[-1].b_res.grad.abs().max() < 1e-6, "a uniform read-out must starve the final mixer"


@pytest.mark.parametrize("mode,n", [("expand", 2), ("expand", 4), ("frac", 4)])
def test_gradients_reach_every_spine_parameter(mode, n):
    _, out, expand, spine = _run(mode, n, lss=True, y_scale=0.1)
    out.square().mean().backward()
    for site in spine:
        for name, p in site.named_parameters():
            if name in ("alpha", "gen_norm.weight"):
                continue  # both are gated by phi, which is zero-initialised; they move once it does
            assert p.grad is not None and p.grad.abs().max() > 0, f"{mode}/{name} got no gradient"
    assert expand.scale.grad.abs().max() > 0


def test_frac_keeps_the_residual_the_same_size_and_expand_multiplies_it():
    h = torch.randn(2, 32, 64)
    assert StreamExpand(64, 4, "frac")(h).shape == (2, 32, 4, 16)      # 4 x 16 = 64, unchanged
    assert StreamExpand(64, 4, "expand")(h).shape == (2, 32, 4, 64)    # 4 x 64, four times the memory


def test_frac_refuses_a_width_it_cannot_divide():
    with pytest.raises(ValueError, match="frac needs"):
        Spine(66, 4, site_idx=0, mode="frac")


def test_spine_parameters_are_kept_away_from_muon_and_weight_decay():
    """A 4x4 doubly-stochastic bias put through Newton-Schulz orthogonalisation fights its own
    manifold, and `split_muon_params` routes on shape alone."""
    site = Spine(64, 4, site_idx=0)
    for name, p in site.named_parameters():
        if name == "phi.weight":
            assert getattr(p, "_no_muon", False) and getattr(p, "_no_reinit", False)
        elif p.ndim >= 1 and name.startswith("b_") or name == "alpha":
            assert getattr(p, "_no_muon", False) and getattr(p, "_no_weight_decay", False)


def test_generator_columns_match_the_published_parameter_counts():
    assert n_cols("spectral_sphere", 4) == 9      # 2·k + (n−1) with k = (n−1)(n−2)/2 = 3
    assert n_cols("orthogonal", 4) == 3
    assert n_cols("sinkhorn", 4) == 16
    assert n_cols("perm_convex", 4) == 24         # n! — mHC-lite's factorial cost
    assert n_cols("spectral_sphere", 2) == 1      # identity ↔ swap, and everything between


# --- wired into the model -------------------------------------------------------------------------
import dataclasses  # noqa: E402

from mote.config import MoteConfig, SpineCfg  # noqa: E402
from mote.model.hnet import HNetForCausalLM  # noqa: E402


def _model(mode="off", n=4, project="spectral_sphere", preset="mote_1m", seed=3, **kw):
    cfg = dataclasses.replace(getattr(MoteConfig, preset)(), spine=SpineCfg(mode=mode, n=n, project=project, **kw))
    torch.manual_seed(seed)
    return cfg, HNetForCausalLM(cfg).eval()


def test_the_spine_is_off_by_default_and_leaves_the_model_exactly_as_it_was():
    cfg, model = _model("off")
    assert cfg.spine.mode == "off" and not model.spine_on
    assert model.residual_proj is not None                    # the zero-init skip is still the skip
    assert model.encoder.spines is None and model.decoder.spines is None
    assert not hasattr(model, "stream") and not hasattr(model, "chunk_spine")


@pytest.mark.parametrize("mode", ["expand", "frac"])
def test_the_spine_has_seven_sites_and_subsumes_the_zero_init_skip(mode):
    cfg, model = _model(mode, preset="mote_96m")
    assert model.residual_proj is None, "the chunk stage's H_post is the write path now"
    sites = len(model.encoder.spines) + 1 + len(model.decoder.spines)
    assert sites == cfg.encoder_layers + 1 + cfg.decoder_layers == 7
    # HC's rotating read only breaks the stream symmetry if it counts across the whole spine
    idx = [s.site_idx for s in model.encoder.spines] + [model.chunk_spine.site_idx] + [s.site_idx for s in model.decoder.spines]
    assert idx == list(range(7)), idx


@pytest.mark.parametrize("mode,project", [
    ("off", "spectral_sphere"), ("expand", "spectral_sphere"), ("frac", "spectral_sphere"),
    ("expand", "orthogonal"), ("expand", "sinkhorn"), ("expand", "perm_convex"),
    ("expand", "diag"), ("expand", "none"),
])
def test_prefill_and_step_agree_with_forward(mode, project):
    """Four separate byte paths now branch on the spine; a divergence here is a serving bug that
    would only ever show up as a model that trains fine and generates garbage."""
    _, model = _model(mode, project=project)
    ids = torch.randint(0, 200, (1, 80))
    with torch.no_grad():
        full = model(ids)
        state = model.allocate_inference_state(torch.device("cpu"))
        pre = model.prefill(ids[:, :64], state)
        stepped = torch.cat([model.step(ids[:, t:t + 1], state)[0] for t in range(64, 80)], 1)
    # subtract first, then drop the padded rows: the head masks ids >= vocab_size to -inf, and
    # (-inf) - (-inf) is nan, so selecting before subtracting would compare nothing at all.
    finite = torch.isfinite(full.logits)
    assert (pre.logits - full.logits[:, :64])[finite[:, :64]].abs().max() < 3e-4
    assert (stepped - full.logits[:, 64:])[finite[:, 64:]].abs().max() < 3e-4


@pytest.mark.parametrize("mode", ["expand", "frac"])
def test_a_spine_model_trains_end_to_end(mode):
    _, model = _model(mode)
    model.train()
    out = model(torch.randint(0, 200, (2, 96)))
    (out.logits[torch.isfinite(out.logits)].square().mean()).backward()
    for name, p in model.named_parameters():
        if p.requires_grad and ("spine" in name or "stream" in name):
            if name.endswith(("alpha", "gen_norm.weight")):
                continue  # gated by the zero-initialised generator until it moves
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_the_spine_survives_a_config_round_trip(tmp_path):
    cfg, _ = _model("frac", n=2, project="orthogonal")
    cfg.save(tmp_path / "config.json")
    back = MoteConfig.load(tmp_path / "config.json")
    assert back.spine == cfg.spine
    assert HNetForCausalLM(back).spine_on
