"""Serving reads a prompt in windows (`prefill_window`), so prefill(x[:k]) + forward_from_state(x[k:]) must equal
prefill(x) for EVERY split k — including splits right after a boundary, inside the dechunk EMA's carry, at
Mamba-3 reference-window edges and with the trunk's rate floor on (the floor scales with the window, so a
64-byte continuation gets the same bytes-per-chunk bound as the 16384-byte training window)."""

import pytest
import torch

import mote.model.mamba3 as mamba3
from mote.config import RelationCfg
from mote.model.hnet import HNetForCausalLM
from conftest import tiny_cfg


def _model(seed, floor=0):
    torch.manual_seed(seed)
    cfg = tiny_cfg(encoder_layers=2, main=RelationCfg(n_layers=2, d_model=32, n_heads=2, d_ff=64), max_seq_len=16384)
    cfg.dc.bound_floor = floor
    return HNetForCausalLM(cfg, dtype=torch.float32).eval()


def _finite_rel(a, b):
    ok = torch.isfinite(a) & torch.isfinite(b)
    return ((a[ok] - b[ok]).norm() / a[ok].norm().clamp_min(1e-9)).item()


@pytest.mark.parametrize("floor", [0, 2048])
@torch.no_grad()
def test_every_split_matches_one_shot_prefill(floor, monkeypatch):
    monkeypatch.setattr(mamba3, "REF_CHUNK", 16)
    g = torch.Generator().manual_seed(5)
    model = _model(0, floor)
    x = torch.randint(0, 256, (1, 200), generator=g)
    st = model.allocate_inference_state("cpu")
    full = model.prefill(x, st)
    bm_full = full.routing.boundary_mask[0]
    for k in list(range(1, 200, 7)) + [63, 64, 65, 127, 128, 129, 199]:
        s = model.allocate_inference_state("cpu")
        a = model.prefill(x[:, :k], s)
        lg, bm, _ = model.forward_from_state(x[:, k:], s)
        assert torch.equal(torch.cat([a.routing.boundary_mask[0], bm]), bm_full), (floor, k)
        assert _finite_rel(torch.cat([a.logits, lg], 1), full.logits) < 1e-4, (floor, k)
        assert s.main.n == st.main.n, (floor, k)
