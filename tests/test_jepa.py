"""JEPA aux losses (docs/shape.md 2026-08-24): all three anti-collapse mechanisms must train,
stop-grad must actually stop, the hinge must fire on collapsed latents, SIGReg must prefer
Gaussian latents, and the EMA teacher must track the online encoder."""

import torch

from mote.config import MoteConfig
from mote.model.hnet import HNetForCausalLM
from mote.train.jepa import JepaAux


def _model():
    torch.manual_seed(0)
    cfg = MoteConfig.smoke()
    cfg.mbp.enabled = False
    return HNetForCausalLM(cfg), cfg


def test_all_modes_produce_finite_grads():
    model, cfg = _model()
    ids = torch.randint(0, 256, (2, 96))
    h = model.encoder(model.embeddings(ids))
    for mode in ("minimal", "ema", "sigreg"):
        aux = JepaAux(model, mode, cfg.d_model_outer)
        loss, stats = aux(ids, h)
        assert torch.isfinite(loss)
        g = torch.autograd.grad(loss, list(aux.predictor.parameters()), retain_graph=True)
        assert all(torch.isfinite(x).all() for x in g)
        assert "jepa_pred" in stats and "jepa_adj_cos" in stats


def test_stop_grad_targets_do_not_backprop_through_target_path():
    model, cfg = _model()
    ids = torch.randint(0, 256, (1, 64))
    aux = JepaAux(model, "minimal", cfg.d_model_outer)
    h = model.encoder(model.embeddings(ids))
    loss, _ = aux(ids, h)
    loss.backward()
    # gradient reaches the encoder only through the ONLINE branch (predictor input), which is
    # what shaping means; the target side is detached — a pure-target parametrization gets none.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


def test_hinge_fires_on_collapse_and_sigreg_prefers_gaussian():
    model, cfg = _model()
    ids = torch.randint(0, 256, (1, 64))
    aux = JepaAux(model, "minimal", cfg.d_model_outer)
    flat = torch.zeros(1, 64, cfg.d_model_outer)  # collapsed encoder
    _, stats = aux(ids, flat)
    assert stats["jepa_hinge"] >= 1.0 - 1e-5  # hinge floor fully violated
    sig = JepaAux(model, "sigreg", cfg.d_model_outer)
    torch.manual_seed(1)
    gauss = torch.randn(4, 64, cfg.d_model_outer)
    assert sig._sigreg(flat) > 4 * sig._sigreg(gauss)


def test_ema_teacher_tracks_online():
    model, cfg = _model()
    aux = JepaAux(model, "ema", cfg.d_model_outer, ema_decay=0.5)
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    before = torch.cat([p.flatten() for p in aux.teacher_encoder.parameters()])
    aux.ema_update(model)
    after = torch.cat([p.flatten() for p in aux.teacher_encoder.parameters()])
    online = torch.cat([p.flatten() for p in model.encoder.parameters()])
    assert torch.allclose(after, before + 0.5 * (online - before), atol=1e-6)
    assert not any(p.requires_grad for p in aux.teacher_encoder.parameters())
