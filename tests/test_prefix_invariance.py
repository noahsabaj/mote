"""Prefix-invariance audit (2608.22876, adopted 2026-08-29): two forwards that differ only at the last byte
must agree at every earlier position in every layer. Attention-mask inspection cannot see a leak that
comes from a reduction along the wrong axis or a global ranking; this can, and it says WHICH module.

The audit is a structural property, so it runs on random weights (mote-1m and a tiny hybrid) on the CPU
reference paths with the Mamba-3 reference window shrunk so inter-window carries are exercised at a short
length. Two gates from the paper: a clean verdict counts only if an injected shift fault is localised to
the module it was injected into (positive control), and the window must exceed every internal block size
in its own units (Mamba-3 chunk 64 bytes, dechunk EMA block 64 chunks, bound_min_len 64).

It also records the one non-causal path Mote has by construction: `project_boundaries` ranks the whole
window, so when the floor or ceiling BINDS a boundary decision can depend on later bytes (dc.py `_bound`).
The floor never binds at the natural rate — this test pins both halves of that statement.
"""

from typing import Dict, Tuple

import pytest
import torch

import mote.model.mamba3 as mamba3
from mote.config import Mamba3Cfg, RelationCfg, resolve_preset
from mote.model.blocks import Block
from mote.model.dc import DeChunkLayer, RoutingModule, project_boundaries
from mote.model.hnet import HNetForCausalLM
from conftest import tiny_cfg

T = 2048
S = 64  # the changed suffix: long enough to contain a chunk boundary, so the main network's last chunks differ too


def _named_modules(model: HNetForCausalLM):
    names = {}
    for stack, tag in ((model.encoder, "encoder"), (model.main_network, "main"), (model.decoder, "decoder")):
        for i, b in enumerate(stack.layers):
            names[b] = f"{tag}.{i}"
    names[model.routing_module] = "routing"
    names[model.dechunk_layer] = "dechunk"
    names[model.lm_head] = "lm_head"
    return names


def _run(model: HNetForCausalLM, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], int]:
    """Every audited module's output for one forward, plus the number of chunks whose boundary byte lies
    before the changed suffix (those chunks' inputs are identical between the two runs)."""
    names = _named_modules(model)
    outs: Dict[str, torch.Tensor] = {}

    def hook(mod, inp, out):
        name = names[mod]
        if isinstance(mod, Block):
            outs[name] = out[0].detach().float()  # (hidden, residual, cache)
        elif isinstance(mod, RoutingModule):
            outs[name] = out.boundary_prob[..., 1].detach().float()
        else:
            outs[name] = out.detach().float()

    handles = [m.register_forward_hook(hook) for m in names]
    try:
        with torch.no_grad():
            out = model(x)
    finally:
        for h in handles:
            h.remove()
    return outs, int(out.routing.boundary_mask[0, : T - S].sum())


def audit(model: HNetForCausalLM, x: torch.Tensor) -> Dict[str, float]:
    """max |Δ| per module over the shared prefix between `x` and `x` with its last S bytes changed. Byte-level
    modules compare positions [0, T-S); the main network's chunk-level outputs compare the chunks whose
    boundary byte lies before T-S (the same chunks in both runs — the router is causal there too)."""
    x2 = x.clone()
    x2[0, -S:] = (x2[0, -S:] + 101) % 256
    a, na = _run(model, x)
    b, nb = _run(model, x2)
    assert na == nb, (na, nb)  # boundaries before the suffix agree, or the router itself leaked
    deltas = {}
    for name in a:
        if name.startswith("main."):
            deltas[name] = float((a[name][:, :na] - b[name][:, :na]).abs().max())
        else:
            deltas[name] = float((a[name][:, : T - S] - b[name][:, : T - S]).abs().max())
    return deltas


def _first_bad(deltas: Dict[str, float], tau: float = 1e-6):
    order = [n for n in deltas if n.startswith("encoder")] + ["routing"] + [n for n in deltas if n.startswith("main")] + ["dechunk"] + [n for n in deltas if n.startswith("decoder")] + ["lm_head"]
    for n in order:
        if deltas[n] > tau:
            return n
    return None


@pytest.fixture(autouse=True)
def short_mamba_windows(monkeypatch):
    monkeypatch.setattr(mamba3, "REF_CHUNK", 16)  # carries between reference windows at every 16 bytes


def _models():
    torch.manual_seed(0)
    cfg = resolve_preset("mote-1m")
    cfg.dc.bound_floor, cfg.max_seq_len = 2048, 16384  # the trunk's floor: a rate, never binding here
    # One chunk bucket for the whole window: both runs then pad the chunk axis to the same length, so every
    # GEMM has the same shape and a clean prefix is bitwise equal (a different padded length picks different
    # CPU GEMM micro-kernels and leaves ~1e-10 floors that would need the paper's epsilon sweep to dismiss).
    cfg.dc.chunk_bucket = T
    yield "mote-1m", HNetForCausalLM(cfg, device="cpu").float().eval()
    torch.manual_seed(0)
    hcfg = tiny_cfg(main=RelationCfg(n_layers=4, d_model=32, n_heads=2, d_ff=64, pattern="MRMR", out_gate=True, mamba_out_norm=True),
                    mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=T)
    hcfg.dc.chunk_bucket = T
    yield "hybrid", HNetForCausalLM(hcfg, device="cpu").float().eval()


@pytest.mark.parametrize("name,model", list(_models()), ids=lambda v: v if isinstance(v, str) else "")
def test_every_module_is_prefix_invariant_and_the_positive_control_localises(name, model):
    torch.manual_seed(1)
    x = torch.randint(0, 256, (1, T))
    deltas = audit(model, x)
    assert _first_bad(deltas) is None, deltas

    # positive control: a one-position shift injected into a block's hidden output must be localised there
    for stack, idx in (("encoder", 0), ("main", 0), ("decoder", 0)):
        block = getattr(model, {"encoder": "encoder", "main": "main_network", "decoder": "decoder"}[stack]).layers[idx]
        orig = block.forward

        def shifted(*args, _orig=orig, **kw):
            hidden, residual, cache = _orig(*args, **kw)
            return torch.cat([hidden[:, 1:], hidden[:, -1:]], dim=1), residual, cache

        block.forward = shifted
        try:
            bad = _first_bad(audit(model, x))
        finally:
            block.forward = orig
        assert bad == f"{stack}.{idx}", (stack, idx, bad)


def test_the_projection_is_the_one_non_causal_path_and_only_when_it_binds():
    # p at four positions; the floor asks for 2 boundaries. Position 3's confidence decides whether
    # position 2 keeps its boundary — a later byte changing an earlier decision.
    m = torch.ones(1, 4, dtype=torch.bool)
    low = torch.tensor([[1.0, 0.40, 0.45, 0.30]])
    high = torch.tensor([[1.0, 0.40, 0.45, 0.90]])
    b_low, b_high = low > 0.5, high > 0.5
    out_low = project_boundaries(b_low, low, m, torch.tensor([2]), torch.tensor([4]))
    out_high = project_boundaries(b_high, high, m, torch.tensor([2]), torch.tensor([4]))
    assert out_low[0, 2] and not out_high[0, 2]  # the leak, when the floor binds
    # inside the bounds the projection is the identity and the router stays causal
    assert torch.equal(project_boundaries(b_low, low, m, torch.tensor([0]), torch.tensor([4])), b_low)
