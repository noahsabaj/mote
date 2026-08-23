"""Exactness tests for the speed rewrites: block-local LCA attention == dense masked attention;
view-based Givens == index/scatter Givens; cached RoPE == recomputed RoPE."""

import torch
import torch.nn.functional as F

from morpheme.model import mbp as M
from morpheme.model.relation import FullRelation, _rope_cos_sin, rope_tables


def _random_chunks(B: int, L: int, p_boundary: float, seed: int, long_chunk: bool = False) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    b = torch.rand(B, L, generator=g) < p_boundary
    b[:, 0] = True
    if long_chunk:  # one chunk wider than a block, so a query block needs 3+ key blocks
        b[0, 100:240] = False
    return torch.cumsum(b.long(), dim=1) - 1


@torch.no_grad()
def _compare(B, L, p, seed, long_chunk=False, valid_tail=None):
    H, dh = 4, 32
    g = torch.Generator().manual_seed(seed + 1)
    q, k, v = (torch.randn(B, H, L, dh, generator=g) for _ in range(3))
    chunk_id = _random_chunks(B, L, p, seed, long_chunk)
    valid = torch.ones(B, L, dtype=torch.bool)
    if valid_tail is not None:
        valid[:, -valid_tail:] = False
    dense = F.scaled_dot_product_attention(q, k, v, attn_mask=M.lca_mask(chunk_id, valid)[:, None])
    local = M.block_local_attention(q, k, v, chunk_id, valid)
    return (dense - local).abs().max().item()


def test_block_local_matches_dense_typical_chunks():
    assert _compare(2, 256, 0.3, 0) < 1e-5


def test_block_local_matches_dense_long_chunk_and_odd_length():
    assert _compare(2, 2047, 0.25, 1, long_chunk=True) < 1e-5  # L=2047 is what training sees


def test_block_local_matches_dense_sparse_boundaries_and_invalid_tail():
    assert _compare(3, 500, 0.12, 2, valid_tail=37) < 1e-5


def test_block_local_window_lo():
    chunk_id = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 3]])
    lo = M.window_lo(chunk_id)
    # chunk 0 starts 0, chunk 1 at 3, chunk 2 at 5, chunk 3 at 8 -> lo = start of previous chunk
    assert lo.tolist() == [[0, 0, 0, 0, 0, 3, 3, 3, 5]]


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
