"""`GET /api/checkpoints`: the file size it reports, and the cache that keeps it cheap.

Describing one checkpoint costs a torch.load and a full read of the run log, so the rows are
cached (docs/checkpoints.md). The failure mode worth a test is a *stale* cache — a run that
gains its first eval record while the .pt on disk never changes, which is exactly what happens
while training is still going.
"""

import json
import time
from pathlib import Path

import pytest

import mote.serve.app as A


class StubInfo:
    def __init__(self, val_bpb, bytes_seen):
        self.val_bpb, self.bytes_seen = val_bpb, bytes_seen


class StubEngine:
    """Stands in for a loaded Engine: counts how often a checkpoint is actually described."""

    def __init__(self):
        self.calls = 0
        self.ckpt_path = Path("nowhere.pt")

    def _describe_checkpoint(self, path, step, extra):
        self.calls += 1
        log = path.parent / "log.jsonl"
        val = None
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if "eval" in rec:
                    val = rec["eval"]["val_bpb"]
        return StubInfo(val, step * 1024)


@pytest.fixture
def run(tmp_path, monkeypatch):
    d = tmp_path / "runs" / "ab_muon_2048"
    d.mkdir(parents=True)
    ckpt = d / "last.pt"
    ckpt.write_bytes(b"\0" * 4096)
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "torch", type("T", (), {"load": staticmethod(lambda *a, **k: {"step": 2000})}))
    A._CKPT_ROWS.clear()
    return ckpt


def test_row_reports_file_size_and_is_cached(run, monkeypatch):
    eng = StubEngine()
    monkeypatch.setitem(A.STATE, "engine", eng)

    first = A._checkpoint_row(run)
    assert first["file_size_bytes"] == 4096
    assert first["step"] == 2000
    assert first["bytes_seen"] == 2000 * 1024  # bytes seen is training data, not the file
    assert eng.calls == 1

    assert A._checkpoint_row(run) == first
    assert eng.calls == 1, "an unchanged checkpoint must not be described twice"


def test_a_new_eval_record_invalidates_even_though_the_checkpoint_did_not_change(run, monkeypatch):
    eng = StubEngine()
    monkeypatch.setitem(A.STATE, "engine", eng)
    assert A._checkpoint_row(run)["val_bpb"] is None

    log = run.parent / "log.jsonl"
    log.write_text(json.dumps({"step": 2000, "eval": {"val_bpb": 1.761}}) + "\n", encoding="utf-8")
    # Same .pt, same mtime: only the log moved, and the row has to follow it.
    assert A._checkpoint_row(run)["val_bpb"] == 1.761
    assert eng.calls == 2


def test_nothing_is_cached_while_no_engine_can_describe_it(run, monkeypatch):
    monkeypatch.setitem(A.STATE, "engine", None)
    assert A._checkpoint_row(run)["val_bpb"] is None
    assert A._CKPT_ROWS == {}, "caching a null row would keep it null after an engine loads"

    eng = StubEngine()
    monkeypatch.setitem(A.STATE, "engine", eng)
    log = run.parent / "log.jsonl"
    log.write_text(json.dumps({"step": 2000, "eval": {"val_bpb": 0.954}}) + "\n", encoding="utf-8")
    assert A._checkpoint_row(run)["val_bpb"] == 0.954


def test_a_rewritten_checkpoint_is_read_again(run, monkeypatch):
    eng = StubEngine()
    monkeypatch.setitem(A.STATE, "engine", eng)
    A._checkpoint_row(run)
    time.sleep(0.01)
    run.write_bytes(b"\0" * 8192)
    assert A._checkpoint_row(run)["file_size_bytes"] == 8192
    assert eng.calls == 2
