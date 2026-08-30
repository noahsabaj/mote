"""--stop-step / --stop-minutes end a run gracefully without touching its schedule (2026-08-30). The cloud arms
used a shell poller that SIGTERMed a launcher shim; the trainer ran on as an orphan to its full horizon."""

import json
from pathlib import Path

import pytest
import torch

from mote.train.train import Trainer

DATA = "/home/nsabaj/Development/mote/data/local_mix"
pytestmark = pytest.mark.skipif(not Path(DATA + ".train.bin").exists(), reason="needs data/local_mix")
BASE = ["--preset", "mote-1m", "--data", DATA, "--device", "cpu", "--batch-size", "2", "--seq-len", "256",
        "--optimizer", "muon", "--lr", "1e-3", "--schedule", "wsd", "--max-steps", "20", "--log-every", "1",
        "--eval-every", "100000", "--eval-batches", "1", "--ckpt-minutes", "999", "--norm-guard", "off", "--seed", "3"]


def _records(out):
    return [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]


def test_stop_step_ends_the_run_with_the_schedule_intact(tmp_path):
    torch.set_num_threads(1)
    full = Trainer(BASE + ["--out", str(tmp_path / "full")])
    for _ in full.run():
        pass
    short = Trainer(BASE + ["--out", str(tmp_path / "short"), "--stop-step", "6"])
    for _ in short.run():
        pass
    assert short.step == 6 and short.stopped_reason == "stop-step"
    a = {r["step"]: r for r in _records(tmp_path / "full") if "train_bpb" in r}
    b = {r["step"]: r for r in _records(tmp_path / "short") if "train_bpb" in r}
    assert set(b) == set(range(1, 7))
    for s in range(1, 7):  # same lr at every step: the 20-step schedule was not shortened to 6
        assert a[s]["lr"] == b[s]["lr"] and a[s]["train_bpb"] == b[s]["train_bpb"]
    recs = _records(tmp_path / "short")
    assert any(r.get("stopped") == "stop-step" for r in recs)
    done = [r for r in recs if r.get("done")]
    assert done and done[-1]["final_step"] == 6 and done[-1]["interrupted"] is False
    assert any(r.get("eval", {}).get("final") if isinstance(r.get("eval"), dict) else False for r in recs) or any("final" in r for r in recs)
    assert (tmp_path / "short" / "last.pt").exists()
