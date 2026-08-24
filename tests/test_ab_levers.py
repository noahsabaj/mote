"""The two A/B levers for the flagship freeze (docs/shape.md): the Relation chunk window and the
bf16 residual stream. The window must change nothing when it covers the whole sequence, mask exactly
when it does not, and both levers must train."""

import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM
from mote.model.relation import FullRelation


def _cfg(**kw):
    base = dict(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )
    base.update(kw)
    return MoteConfig(**base)


@torch.no_grad()
def test_window_covering_everything_changes_nothing():
    torch.manual_seed(0)
    full = FullRelation(32, 2, layer_idx=0)
    torch.manual_seed(0)
    wide = FullRelation(32, 2, layer_idx=0, window=999)
    x = torch.randn(1, 24, 32)
    assert torch.allclose(full(x), wide(x), atol=1e-6)


@torch.no_grad()
def test_window_masks_the_far_past_exactly():
    torch.manual_seed(0)
    win = FullRelation(32, 2, layer_idx=0, window=4)
    x = torch.randn(1, 24, 32)
    y_all = win(x)
    # position t may draw on positions {t-3..t}: perturbing anything older must not change output at t
    x2 = x.clone()
    x2[0, :10] += 3.0
    y_pert = win(x2)
    assert torch.allclose(y_all[0, 14:], y_pert[0, 14:], atol=1e-5)  # t >= 14 never sees pos < 10... (10+4)
    assert not torch.allclose(y_all[0, 10:13], y_pert[0, 10:13], atol=1e-3)  # near the edit it must change


def test_both_levers_build_and_train_a_step():
    for kw in ({"residual_in_fp32": False}, {}):
        cfg = _cfg(**kw)
        cfg.main.window_chunks = 8 if not kw else None
        torch.manual_seed(0)
        model = HNetForCausalLM(cfg)
        ids = torch.randint(0, 256, (2, 96))
        out = model(ids)
        loss = out.logits.float().pow(2).mean()
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        assert torch.isfinite(loss) and torch.isfinite(g).all()
    rt = MoteConfig.from_dict(cfg.to_dict())
    assert rt.main.window_chunks == cfg.main.window_chunks and rt.residual_in_fp32 == cfg.residual_in_fp32
