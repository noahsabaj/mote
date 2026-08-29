"""Exactness tests for the speed rewrites: view-based Givens == index/scatter Givens; cached RoPE ==
recomputed RoPE."""

import torch

from mote.model.relation import FullRelation, _rope_cos_sin, rope_tables


def _givens_reference(mod: FullRelation, info: torch.Tensor) -> torch.Tensor:
    c = torch.cos(mod.theta).to(info.dtype)[None, :, None, None]
    s = torch.sin(mod.theta).to(info.dtype)[None, :, None, None]
    ia, ib = info[:, mod.pair_a], info[:, mod.pair_b]
    out = info.clone()
    out[:, mod.pair_a] = c * ia - s * ib
    out[:, mod.pair_b] = s * ia + c * ib
    return out


@torch.no_grad()
def test_givens_view_form_matches_scatter_form():
    for layer_idx in (0, 1):
        mod = FullRelation(64, 8, layer_idx=layer_idx)
        mod.theta.copy_(torch.linspace(-1.0, 1.0, 4))
        info = torch.randn(2, 8, 13, 8)
        assert torch.allclose(mod._givens(info), _givens_reference(mod, info), atol=1e-6)


def test_rope_cache_matches_direct():
    cos, sin = rope_tables(5, 17, 16, 10000.0, torch.float32, torch.device("cpu"))
    c2, s2 = _rope_cos_sin(torch.arange(5, 22), 16, 10000.0, torch.float32)
    assert torch.equal(cos, c2) and torch.equal(sin, s2)
    assert rope_tables(5, 17, 16, 10000.0, torch.float32, torch.device("cpu"))[0] is cos  # memoised
