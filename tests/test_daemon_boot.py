"""The daemon and a missing GPU (2026-08-28): it waits for CUDA before anything in the process asks torch
for it, serves on the CPU with the queue paused when CUDA never comes, and lets a person park the engine
on the CPU so a standalone measurement can have the whole card."""

import threading

import torch
from fastapi.testclient import TestClient

import mote.serve.app as A
from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM
from mote.serve.engine import Engine
from mote.serve.jobs import JobQueue


def _fake_probe(answers):
    it = iter(answers)

    def probe():
        return next(it)
    return probe


def test_wait_for_cuda_returns_once_the_probe_says_yes():
    log = []
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += s
        log.append(s)
    why = A.wait_for_cuda(180.0, probe=_fake_probe([(False, "no device"), (False, "no device"), (True, "ok")]),
                          interval=5.0, clock=lambda: clock["t"], sleep=sleep)
    assert why is None and log == [5.0, 5.0]  # two waits, then the yes


def test_wait_for_cuda_gives_up_with_the_last_reason_after_the_timeout():
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += s
    why = A.wait_for_cuda(12.0, probe=_fake_probe([(False, "a")] * 2 + [(False, "b")] * 5), interval=5.0,
                          clock=lambda: clock["t"], sleep=sleep)
    assert why == "b" and clock["t"] == 12.0  # 5 + 5 + 2: the last sleep is clipped to the deadline


def test_the_real_probe_asks_a_fresh_interpreter():
    ok, why = A.cuda_probe()
    assert ok == torch.cuda.is_available() and isinstance(why, str)


def _tiny_engine():
    cfg = MoteConfig(d_model_outer=32, encoder_layers=1, decoder_layers=1,
                     main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
                     mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3),
                     mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256)
    torch.manual_seed(0)
    return Engine.from_model(HNetForCausalLM(cfg), cfg, device="cpu")


def test_parking_the_engine_keeps_it_on_the_cpu_across_idle_signals(tmp_path, monkeypatch):
    moves = []
    monkeypatch.setattr(A, "_move_engine", lambda device, warm: moves.append((device, warm)))
    monkeypatch.setitem(A.STATE, "engine", _tiny_engine())
    monkeypatch.setitem(A.STATE, "jobs", JobQueue(tmp_path / "jobs.json", threading.Lock()))
    monkeypatch.setitem(A.STATE, "device", "cuda")
    monkeypatch.setitem(A.STATE, "parked", False)
    monkeypatch.setitem(A.STATE, "cuda_missing", None)
    c = TestClient(A.app)
    assert A._serve_device() == "cuda"
    r = c.post("/api/engine/device", json={"device": "cpu"})
    assert r.status_code == 200 and r.json()["parked"] is True and moves == [("cpu", False)]
    assert A._serve_device() == "cpu"
    A._queue_idle()  # the queue going idle used to bring the engine back to the GPU unconditionally
    assert moves[-1] == ("cpu", True)
    r = c.post("/api/engine/device", json={"device": "cuda"})
    assert r.status_code == 200 and r.json()["parked"] is False and moves[-1] == ("cuda", True)
    assert c.post("/api/engine/device", json={"device": "tpu"}).status_code == 400


def test_without_cuda_the_engine_cannot_be_asked_onto_the_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "_move_engine", lambda device, warm: None)
    monkeypatch.setitem(A.STATE, "engine", _tiny_engine())
    monkeypatch.setitem(A.STATE, "jobs", JobQueue(tmp_path / "jobs.json", threading.Lock()))
    monkeypatch.setitem(A.STATE, "device", "cuda")
    monkeypatch.setitem(A.STATE, "cuda_missing", "no device")
    assert A._serve_device() == "cpu"
    r = TestClient(A.app).post("/api/engine/device", json={"device": "cuda"})
    assert r.status_code == 409 and "no device" in r.json()["detail"]
