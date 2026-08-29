"""A checkpoint that carries the trainer's EMA is served and scored on the EMA (the number its gates were read on),
not on the raw weights — found 2026-08-29: the engine and mote.eval.val_bpb loaded ck["model"]."""

import torch

from mote.infer.engine import Engine
from mote.model.hnet import load_weights
from conftest import tiny_model, write_ckpt


def _ckpt_with_ema(tmp_path, seed=0):
    model = tiny_model(seed=seed)
    ema = [p.detach().clone() + 0.5 for p in model.parameters()]  # visibly different from the raw weights
    path = write_ckpt(tmp_path / "run" / "last.pt", model, step=7, extra={"ema": ema})
    return model, ema, path


def test_load_weights_prefers_the_ema_and_can_be_told_not_to(tmp_path):
    model, ema, path = _ckpt_with_ema(tmp_path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    fresh = tiny_model(seed=1)
    assert load_weights(fresh, ck) == "ema"
    assert all(torch.equal(p, e) for p, e in zip(fresh.parameters(), ema))
    assert load_weights(fresh, ck, prefer_ema=False) == "raw"
    assert all(torch.equal(p, q) for p, q in zip(fresh.parameters(), model.parameters()))


def test_engine_serves_the_ema(tmp_path):
    _, ema, path = _ckpt_with_ema(tmp_path)
    eng = Engine(path, device="cpu")
    assert eng.weights == "ema"
    assert all(torch.equal(p.cpu(), e) for p, e in zip(eng.model.parameters(), ema))


def test_checkpoint_without_ema_serves_raw(tmp_path):
    model = tiny_model(seed=2)
    path = write_ckpt(tmp_path / "run" / "last.pt", model, step=1)
    eng = Engine(path, device="cpu")
    assert eng.weights == "raw"
    assert all(torch.equal(p.cpu(), q) for p, q in zip(eng.model.parameters(), model.parameters()))
