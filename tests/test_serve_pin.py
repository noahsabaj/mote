"""Serving policy + device (signed 2026-08-25, docs/results/2026-08-24-web-qa.md): the pin in .mote/config.json
is the boot default; a finished job replaces what is served only when it was on the air; the engine belongs on
the CPU while any job runs and on the configured device when the queue idles; a manual load takes the running
job off the air."""

import json
from types import SimpleNamespace

import pytest

import mote.serve.app as A


class _Jobs:
    def __init__(self, current=None, runnable=False):
        self.cur, self.runnable, self.set_calls = current, runnable, []

    def current(self):
        return self.cur

    def has_runnable(self):
        return self.runnable

    def set_serve(self, job_id, on):
        self.set_calls.append((job_id, on))
        if self.cur is not None:
            self.cur.serve = on
        return self.cur

    def status(self):
        return {"current": None, "phase": None, "queued": [], "recent": []}


class _Eng:
    def __init__(self, device="cpu"):
        self.device = SimpleNamespace(type=device)

    def info(self):
        return {"checkpoint": {"path": "x"}}


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setitem(A.STATE, "config_file", tmp_path / ".mote" / "config.json")
    monkeypatch.setitem(A.STATE, "device", "cuda")
    monkeypatch.setitem(A.STATE, "jobs", None)
    monkeypatch.setitem(A.STATE, "engine", _Eng("cuda"))
    monkeypatch.setitem(A.STATE, "gate", None)
    monkeypatch.setitem(A.STATE, "swapping", False)
    return tmp_path


def test_pin_round_trips_and_keeps_the_rest_of_the_config(state):
    f = state / ".mote" / "config.json"
    f.parent.mkdir()
    f.write_text(json.dumps({"port": 7861, "checkpoint": "runs/old/last.pt"}), encoding="utf-8")
    run = state / "runs" / "new"
    run.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"\0")
    A._write_pin(run / "last.pt")
    assert json.loads(f.read_text(encoding="utf-8")) == {"port": 7861, "checkpoint": "runs/new/last.pt"}
    assert A._pin_path() == "runs/new/last.pt"


def test_pin_path_is_none_without_a_config(state):
    assert A._pin_path() is None


def test_serve_device_follows_the_queue(state, monkeypatch):
    assert A._serve_device() == "cuda"  # no queue at all
    monkeypatch.setitem(A.STATE, "jobs", _Jobs())
    assert A._serve_device() == "cuda"  # idle queue
    monkeypatch.setitem(A.STATE, "jobs", _Jobs(runnable=True))
    assert A._serve_device() == "cpu"  # about to start
    monkeypatch.setitem(A.STATE, "jobs", _Jobs(current=SimpleNamespace(id="j1", serve=False, out_dir="runs/a")))
    assert A._serve_device() == "cpu"  # running


def test_only_a_job_on_the_air_replaces_the_served_model(state, monkeypatch):
    run = state / "runs" / "arm"
    run.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"\0")
    loads = []
    monkeypatch.setattr(A, "_load_engine", lambda path: loads.append(path) or _Eng("cuda"))
    before = A.STATE["engine"]
    A._job_finished(SimpleNamespace(state="done", serve=False, out_dir="runs/arm", id="j1"))
    assert loads == [] and A.STATE["engine"] is before and A._pin_path() is None  # a screening arm: untouched
    A._job_finished(SimpleNamespace(state="failed", serve=True, out_dir="runs/arm", id="j2"))
    assert loads == [] and A._pin_path() is None  # a failed job pins nothing
    A._job_finished(SimpleNamespace(state="done", serve=True, out_dir="runs/arm", id="j3"))
    assert loads == [run / "last.pt"] and A.STATE["engine"] is not before
    assert A._pin_path() == "runs/arm/last.pt"


def test_manual_load_pins_and_takes_the_running_job_off_the_air(state, monkeypatch):
    run = state / "runs" / "pick"
    run.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"\0")
    cur = SimpleNamespace(id="j9", serve=True, out_dir="runs/trunk")
    jobs = _Jobs(current=cur)
    monkeypatch.setitem(A.STATE, "jobs", jobs)
    monkeypatch.setattr(A, "_load_engine", lambda path: _Eng("cpu"))
    monkeypatch.setattr(A, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(A, "model_info", lambda: {"checkpoint": {"path": "runs/pick/last.pt"}})
    info = A.load_checkpoint(SimpleNamespace(id="runs/pick/last.pt"))
    assert info["unfollowed"] == "runs/trunk" and jobs.set_calls == [("j9", False)] and cur.serve is False
    assert A._pin_path() == "runs/pick/last.pt" and A.STATE["engine"].device.type == "cpu"
    # the running job is a plain arm: nothing to unfollow
    cur2 = SimpleNamespace(id="j10", serve=False, out_dir="runs/arm")
    monkeypatch.setitem(A.STATE, "jobs", _Jobs(current=cur2))
    assert "unfollowed" not in A.load_checkpoint(SimpleNamespace(id="runs/pick/last.pt"))


def test_serve_endpoint(state, monkeypatch):
    cur = SimpleNamespace(id="j1", serve=False, out_dir="runs/trunk")
    jobs = _Jobs(current=cur)
    monkeypatch.setitem(A.STATE, "jobs", jobs)
    A.training_serve(A.TrainServeBody(id=None, on=True))
    assert jobs.set_calls == [(None, True)] and cur.serve is True
    monkeypatch.setitem(A.STATE, "jobs", _Jobs())
    with pytest.raises(A.HTTPException):
        A.training_serve(A.TrainServeBody(id="nope", on=True))


def test_model_info_reports_pin_device_and_following(state, monkeypatch):
    cur = SimpleNamespace(id="j1", serve=True, out_dir="runs/trunk")
    monkeypatch.setitem(A.STATE, "jobs", _Jobs(current=cur))
    monkeypatch.setitem(A.STATE, "engine", _Eng("cpu"))
    monkeypatch.setattr(A, "challenger_info", lambda: None)
    info = A.model_info()
    assert info["pin"] is None and info["serving_device"] == "cpu" and info["following"] == "runs/trunk"
    cur.serve = False
    assert A.model_info()["following"] is None
