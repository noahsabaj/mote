"""The causal-attention ablation control (RelationCfg.mixer = "attention") must be
parameter-matched to FullRelation, causal, and train end-to-end through the H-Net."""

import torch
import torch.nn.functional as F

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.attention import CausalAttention
from mote.model.hnet import HNetForCausalLM
from mote.model.relation import FullRelation


def test_parameter_matched_to_relation():
    rel = FullRelation(64, 4, layer_idx=0)
    attn = CausalAttention(64, 4, layer_idx=0)
    n_rel = sum(p.numel() for p in rel.parameters())
    n_attn = sum(p.numel() for p in attn.parameters())
    assert n_attn == 4 * 64 * 64
    assert 0 <= n_rel - n_attn <= 8  # Relation adds only lambda + H/2 Givens angles


@torch.no_grad()
def test_causal_and_cache_consistent():
    torch.manual_seed(0)
    m = CausalAttention(64, 4, layer_idx=0)
    x = torch.randn(1, 24, 64)
    y = m(x)
    x2 = x.clone()
    x2[0, 12:] += 3.0  # future edits must not change the past
    assert torch.allclose(m(x2)[0, :12], y[0, :12], atol=1e-5)
    y_pre, cache = m(x[:, :10], return_cache=True)
    y_rest = m(x[:, 10:], cache=cache)
    assert torch.allclose(torch.cat([y_pre, y_rest], 1), y, atol=1e-4)
    _, g = m(x, return_gates=True)
    assert torch.all(g[:, :, 0] == 0) and torch.all(g[:, :, 1:] > 0) and torch.all(g < 1)


def test_hnet_trains_with_attention_main():
    cfg = MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64, mixer="attention"),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    assert type(model.main_network.layers[0].mixer) is CausalAttention
    ids = torch.randint(0, 256, (2, 96))
    out = model(ids[:, :-1])
# Next-byte cross-entropy, the loss the model is actually trained with (train.py `_masked_ce_sum`).
    # NOT logits.pow(2): the head masks the padding columns to -inf (vocab_size 266 -> pad_vocab_to 272,
    # config.py) so a byte that does not exist can never be sampled, and (-inf)^2 is inf. cross_entropy's
    # log_softmax gives those columns probability zero, which is the whole point of the mask.
    loss = F.cross_entropy(out.logits.float().flatten(0, 1), ids[:, 1:].reshape(-1))
    loss.backward()
    g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    assert torch.isfinite(loss) and torch.isfinite(g).all()
    rt = MoteConfig.from_dict(cfg.to_dict())
    assert rt.main.mixer == "attention"
