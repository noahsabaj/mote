"""The --eval-ema average is zero-started and bias-corrected at every read (2026-08-30). Started at the init
weights, a 0.9999 average carried them for ~50k steps (0.37 at 10k): the flagship lr arms' val_bpb_ema read
5.7–12.9 at step 5,000 against raw 1.9 and was no read until ~50k, and the engine would have served that model.
Pinned here: the corrected average is the float64 reference from the per-step weights; the engine loads, bitwise,
the weights the trainer's own eval scored; a resume carries the step count; a checkpoint from before (no
`ema_zero_init`) keeps its init-started average uncorrected through a resume; a zero-started average with no
update is not served. The serving shadow's correction is pinned in tests/test_jobs.py."""

from pathlib import Path

import pytest
import torch

from mote.model.hnet import HNetForCausalLM, ema_scale, load_weights
from mote.train.train import Trainer
from conftest import tiny_model, write_ckpt

DATA = "/home/nsabaj/Development/mote/data/local_mix"
pytestmark = pytest.mark.skipif(not Path(DATA + ".train.bin").exists(), reason="needs data/local_mix")
DECAY = 0.5
BASE = ["--preset", "mote-1m", "--data", DATA, "--device", "cpu", "--batch-size", "2", "--seq-len", "256",
        "--optimizer", "muon", "--lr", "1e-3", "--schedule", "wsd", "--max-steps", "4", "--log-every", "1",
        "--eval-every", "2", "--eval-batches", "1", "--ckpt-minutes", "999", "--norm-guard", "off", "--seed", "3",
        "--eval-ema", str(DECAY)]


def _run(out, extra=(), stop_at=None):
    """Run the trainer; return it and the parameters after every optimizer step (float64 copies). The first
    yield after a step's optimizer update is a raw-weight moment: the EMA eval swaps its weights in later."""
    t = Trainer(BASE + ["--out", str(out)] + list(extra))
    snaps, seen = {}, t.step
    for _ in t.run():
        if t.step > seen:
            seen = t.step
            snaps[seen] = [p.detach().double().clone() for p in t.model.parameters()]
        if stop_at is not None and t.step >= stop_at and not t._stop:
            t.request_stop("interrupted")
    if t.step not in snaps:
        snaps[t.step] = [p.detach().double().clone() for p in t.model.parameters()]
    return t, snaps


def _reference(snaps, n):
    """Σ_k (1−β) β^(n−k) θ_k / (1 − β^n) in float64."""
    out = None
    for k in range(1, n + 1):
        w = (1 - DECAY) * DECAY ** (n - k)
        out = [w * p for p in snaps[k]] if out is None else [o + w * p for o, p in zip(out, snaps[k])]
    return [o / (1 - DECAY ** n) for o in out]


def _fresh(t):
    return HNetForCausalLM(t.cfg, device=torch.device("cpu"))


def test_corrected_average_is_the_reference_and_the_engine_loads_what_the_eval_scored(tmp_path):
    torch.set_num_threads(1)
    t, snaps = _run(tmp_path / "run")
    assert t.step == 4 and t.ema_steps == 4 and t.ema_zero_init
    ck = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=True)
    ex = ck["extra"]
    assert ex["ema_steps"] == 4 and ex["ema_zero_init"] is True and ex["ema_decay"] == DECAY
    fresh = _fresh(t)
    assert load_weights(fresh, ck) == "ema"
    ref = _reference(snaps, 4)
    for p, r in zip(fresh.parameters(), ref):
        assert torch.allclose(p.double(), r, rtol=1e-5, atol=1e-6)
    # the stored (uncorrected) average is 1 − β⁴ = 15/16 of the reference: served as it is, it would be off
    assert not all(torch.allclose(e.double(), r, rtol=1e-3, atol=1e-6) for e, r in zip(ex["ema"], ref))
    t._swap_in_ema()  # what val_bpb_ema was scored on
    assert all(torch.equal(p, q) for p, q in zip(t.model.parameters(), fresh.parameters()))


def test_resume_carries_the_step_count(tmp_path):
    torch.set_num_threads(1)
    _run(tmp_path / "straight")
    part, _ = _run(tmp_path / "resumed", stop_at=2)
    assert part.step == 2 and part.ema_steps == 2
    cont, _ = _run(tmp_path / "resumed", extra=["--resume"])
    assert cont.step == 4 and cont.ema_steps == 4 and cont.ema_zero_init
    a = torch.load(tmp_path / "straight" / "last.pt", map_location="cpu", weights_only=True)
    b = torch.load(tmp_path / "resumed" / "last.pt", map_location="cpu", weights_only=True)
    assert b["extra"]["ema_steps"] == 4
    assert all(torch.equal(x, y) for x, y in zip(a["extra"]["ema"], b["extra"]["ema"]))


def test_a_checkpoint_from_before_keeps_its_uncorrected_average_through_a_resume(tmp_path):
    torch.set_num_threads(1)
    part, _ = _run(tmp_path / "run", stop_at=2)
    path = tmp_path / "run" / "last.pt"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("ema_steps", "ema_zero_init", "ema_decay"):  # what a checkpoint from before 2026-08-30 carries
        del ck["extra"][k]
    torch.save(ck, path)
    fresh = _fresh(part)
    assert ema_scale(ck["extra"]) == 1.0 and load_weights(fresh, ck) == "ema"
    assert all(torch.equal(p, e) for p, e in zip(fresh.parameters(), ck["extra"]["ema"]))  # as it is
    cont, _ = _run(tmp_path / "run", extra=["--resume"])
    assert cont.step == 4 and cont.ema_zero_init is False
    ck2 = torch.load(path, map_location="cpu", weights_only=True)
    assert ck2["extra"]["ema_zero_init"] is False
    assert load_weights(fresh, ck2) == "ema"
    assert all(torch.equal(p, e) for p, e in zip(fresh.parameters(), ck2["extra"]["ema"]))


def test_a_zero_started_average_with_no_update_is_not_served(tmp_path):
    model = tiny_model(seed=0)
    zeros = [torch.zeros_like(p) for p in model.parameters()]
    path = write_ckpt(tmp_path / "run" / "last.pt", model, step=0,
                      extra={"ema": zeros, "ema_steps": 0, "ema_zero_init": True, "ema_decay": 0.9})
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ema_scale(ck["extra"]) is None
    fresh = tiny_model(seed=1)
    assert load_weights(fresh, ck) == "raw"
    assert all(torch.equal(p, q) for p, q in zip(fresh.parameters(), model.parameters()))
    assert ema_scale({"ema_zero_init": True, "ema_steps": 1, "ema_decay": 0.9}) == pytest.approx(0.1)
    assert ema_scale({"ema": zeros}) == 1.0  # a checkpoint from before: loaded as it is
