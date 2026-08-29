"""Chunk-count bucketing and activation checkpointing must be bit-neutral (same logits, same grads)."""

import torch

from mote.config import Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM


def _tiny(bucket: int) -> MoteConfig:
    cfg = MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        mamba3=Mamba3Cfg(d_state=16, expand=2, headdim=16),
        main=RelationCfg(n_layers=2, d_model=32, n_heads=2, d_ff=64),
    )
    cfg.dc.chunk_bucket = bucket
    return cfg


def _run(cfg, seed=0, ckpt=False):
    torch.manual_seed(seed)
    m = HNetForCausalLM(cfg, dtype=torch.float32)
    m.main_network.grad_checkpoint = ckpt
    ids = torch.randint(0, 256, (2, 96), generator=torch.Generator().manual_seed(1))
    out = m(ids)
    loss = out.logits.float().logsumexp(-1).mean()
    loss.backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return out, grads, m.chunk_layer.bucket


def test_bucketing_is_bit_neutral():
    out1, g1, b1 = _run(_tiny(1))
    out64, g64, b64 = _run(_tiny(64))
    assert b1 == 1 and b64 == 64
    # GEMMs of different shapes may reorder reductions: equal up to float rounding, not bit-for-bit
    assert torch.allclose(out1.logits, out64.logits, atol=1e-5, rtol=1e-5), (out1.logits - out64.logits).abs().max()
    assert torch.equal(out1.chunk_id, out64.chunk_id)
    for n in g1:
        assert torch.allclose(g1[n], g64[n], atol=1e-6, rtol=1e-4), (n, (g1[n] - g64[n]).abs().max())


def test_checkpointing_is_bit_neutral():
    out_a, ga, _ = _run(_tiny(64), ckpt=False)
    out_b, gb, _ = _run(_tiny(64), ckpt=True)
    assert torch.equal(out_a.logits, out_b.logits)
    for n in ga:
        assert torch.allclose(ga[n], gb[n], atol=1e-6, rtol=1e-4), (n, (ga[n] - gb[n]).abs().max())
