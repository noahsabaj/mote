"""A run interrupted and resumed must be the same run: the trunk's daily snapshots, its mix-B continuation and
every daemon restart go through `--resume`. Twenty steps straight versus ten, an interruption (checkpoint only,
no final eval — what the daemon's preemption does) and ten more must agree bit for bit on the CPU, which
pins the data order, the generator, the optimizer state, the EMA and the schedule's clock together."""

import json
import os
from pathlib import Path

import pytest
import torch

from mote.train.train import Trainer

DATA = "/home/nsabaj/Development/mote/data/local_mix"
BASE = ["--preset", "mote-1m", "--data", DATA, "--device", "cpu", "--batch-size", "2", "--seq-len", "256",
        "--optimizer", "muon", "--lr", "1e-3", "--weight-decay", "0.1", "--schedule", "wsd", "--max-steps", "20",
        "--log-every", "1", "--eval-every", "100000", "--eval-batches", "1", "--ckpt-minutes", "999",
        "--norm-guard", "off", "--eval-ema", "0.99", "--seed", "3"]

pytestmark = pytest.mark.skipif(not Path(DATA + ".train.bin").exists(), reason="needs data/local_mix")


def _records(out: Path):
    recs = {}
    for line in (out / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "train_bpb" in r:
            recs[r["step"]] = r
    return recs


def _run(out: Path, extra=(), stop_at=None):
    t = Trainer(BASE + ["--out", str(out)] + list(extra))
    for _ in t.run():
        if stop_at is not None and t.step >= stop_at and not t._stop:
            t.request_stop("interrupted")
    return t


@pytest.mark.parametrize("aug", [[], ["--aug-noise", "0.05", "--aug-r2l", "0.3", "--aug-offset", "3"]], ids=["plain", "augmented"])
def test_interrupt_and_resume_is_bitwise_the_straight_run(tmp_path, aug):
    torch.set_num_threads(1)
    global BASE
    base = BASE
    BASE = base + aug  # the augmentations draw from the shard's numpy RNG, which the checkpoint must carry too
    try:
        _body(tmp_path)
    finally:
        BASE = base


def _body(tmp_path):
    straight = _run(tmp_path / "straight")
    assert straight.step == 20
    part = _run(tmp_path / "resumed", stop_at=10)
    assert part.step == 10 and (tmp_path / "resumed" / "last.pt").exists()
    import shutil
    shutil.copy(tmp_path / "resumed" / "last.pt", tmp_path / "resumed" / "snap_at10.pt")
    cont = _run(tmp_path / "resumed", extra=["--resume"])
    assert cont.step == 20
    a, b = _records(tmp_path / "straight"), _records(tmp_path / "resumed")
    assert set(a) == set(range(1, 21)) and set(range(11, 21)) <= set(b)
    for s in range(11, 21):
        for k in ("train_bpb", "ce", "grad_norm", "lr", "target_ratio", "bpic"):
            assert a[s][k] == b[s][k], (s, k, a[s][k], b[s][k])
    # the resumed EMA is the straight run's EMA
    if "--aug-noise" in BASE:
        # negative control: without the augmentation RNG in the checkpoint the resumed stream replays from its
        # seed and the run is NOT the straight run — the property this test guards is not vacuous
        import mote.data.loader as loader
        orig = loader.ByteShard.set_rng_state
        loader.ByteShard.set_rng_state = lambda self, st: None
        try:
            import shutil
            shutil.copytree(tmp_path / "resumed", tmp_path / "replayed", ignore=shutil.ignore_patterns("log*.jsonl"))
            torch.save(torch.load(tmp_path / "resumed" / "snap_at10.pt", weights_only=False), tmp_path / "replayed" / "last.pt")
            _run(tmp_path / "replayed", extra=["--resume"])
        finally:
            loader.ByteShard.set_rng_state = orig
        c = _records(tmp_path / "replayed")
        assert any(a[s]["train_bpb"] != c[s]["train_bpb"] for s in range(11, 21))
    ea = torch.load(tmp_path / "straight" / "last.pt", map_location="cpu", weights_only=False)["extra"]["ema"]
    eb = torch.load(tmp_path / "resumed" / "last.pt", map_location="cpu", weights_only=False)["extra"]["ema"]
    assert all(torch.equal(x, y) for x, y in zip(ea, eb))
