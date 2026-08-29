"""The hybrid main (signed 2026-08-29, docs/results/2026-08-29-hybrid-ladder-prereg.md): Mamba-3 blocks among
the Relation blocks of the main network, the Relation output gate, the Mamba-3 pre-gate norm and NoPE
Relation. What is under test is that a hybrid is the same model in every mode — training forward, prefill,
continuation and byte-at-a-time step — and that the pieces the ladder's arms switch on do not change shapes
or break causality.
"""

import torch

from mote.config import Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.blocks import main_pattern
from mote.model.hnet import HNetForCausalLM
from mote.model.mamba3 import Mamba3Mixer
from mote.model.relation import FullRelation
from conftest import tiny_cfg

import pytest


def hybrid_cfg(pattern="MRMR", **main_kw) -> MoteConfig:
    return tiny_cfg(main=RelationCfg(n_layers=len(pattern), d_model=32, n_heads=2, d_ff=64, pattern=pattern, **main_kw),
                    mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return HNetForCausalLM(cfg, device="cpu").float().eval()


def _rel(a, b):
    """Relative difference over the finite entries (the head masks the spare vocab row to -inf)."""
    a, b = a.float(), b.float()
    ok = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[ok], b[ok]
    return ((a - b).norm() / a.norm().clamp_min(1e-9)).item()


def test_pattern_builds_the_right_mixers():
    m = _model(hybrid_cfg("MRMR"))
    kinds = [type(b.mixer) for b in m.main_network.layers]
    assert kinds == [Mamba3Mixer, FullRelation, Mamba3Mixer, FullRelation]
    assert all(b.mlp is not None for b in m.main_network.layers)  # every block keeps its FFN
    assert m.main_network.layers[0].mixer.d_model == 32


def test_pattern_rules():
    with pytest.raises(AssertionError):
        main_pattern(RelationCfg(n_layers=4, pattern="RMMM"))  # must start with M
    with pytest.raises(AssertionError):
        main_pattern(RelationCfg(n_layers=4, pattern="MMMR"))  # >= 2 R
    with pytest.raises(AssertionError):
        main_pattern(RelationCfg(n_layers=4, pattern="MRM"))  # length
    assert main_pattern(RelationCfg(n_layers=3)) == "RRR"


@pytest.mark.parametrize("main_kw", [dict(), dict(out_gate=True, mamba_out_norm=True), dict(rope=False)])
def test_hybrid_prefill_continuation_and_step_match_forward(main_kw):
    cfg = hybrid_cfg("MRMR", **main_kw)
    m = _model(cfg)
    x = torch.randint(0, 256, (1, 96))
    with torch.no_grad():
        full = m(x).logits
        st = m.allocate_inference_state("cpu")
        assert set(st.main.mamba) == {0, 2}
        pre = m.prefill(x[:, :60], st).logits
        cont, _, _ = m.forward_from_state(x[:, 60:], st)
        assert _rel(torch.cat([pre, cont], 1), full) < 1e-4
        st2 = m.allocate_inference_state("cpu")
        m.prefill(x[:, :60], st2)
        outs = []
        for t in range(60, 96):
            lg, _, _ = m.step(x[:, t : t + 1], st2)
            outs.append(lg)
        assert _rel(torch.cat(outs, 1), full[:, 60:]) < 1e-4
        # the Mamba-3 main states advanced in both continuations and agree
        for i in (0, 2):
            for a, b in zip(st.main[i], st2.main[i]):
                assert _rel(a, b) < 1e-4


def test_hybrid_state_clone_and_move_copy_the_mamba_states():
    m = _model(hybrid_cfg("MRMR"))
    st = m.allocate_inference_state("cpu")
    with torch.no_grad():
        m.prefill(torch.randint(0, 256, (1, 40)), st)
    snap = HNetForCausalLM.clone_state(st)
    assert snap.main.arena is st.main.arena and snap.main.n == st.main.n
    for i in (0, 2):
        for a, b in zip(snap.main[i], st.main[i]):
            assert a is not b and torch.equal(a, b)
    moved = HNetForCausalLM.move_state(st, "cpu")
    assert set(moved.main.mamba) == {0, 2}


def test_hybrid_is_causal():
    m = _model(hybrid_cfg("MRMR", out_gate=True, mamba_out_norm=True))
    x = torch.randint(0, 256, (1, 80))
    x2 = x.clone()
    x2[:, 60:] = (x2[:, 60:] + 7) % 256
    with torch.no_grad():
        a, b = m(x).logits, m(x2).logits
    assert torch.allclose(a[:, :59], b[:, :59], atol=1e-5)


def test_out_gate_and_norm_change_parameter_counts_only_where_declared():
    base = _model(hybrid_cfg("MRMR")).num_params()
    gated = _model(hybrid_cfg("MRMR", out_gate=True)).num_params()
    normed = _model(hybrid_cfg("MRMR", mamba_out_norm=True)).num_params()
    assert gated - base == 2 * 32 * 32  # one d_model x d_model gate per Relation layer
    assert normed - base == 2 * 64  # one gain per Mamba-3 main layer over d_inner
