"""Serving beside training (docs/shape.md, signed 2026-08-24): the engine runs replies on its own
high-priority stream and memory pool, holds the GPU gate only when MOTE_SERVE_GATED=1, drains the stream
before a weight swap, and produces the same reply either way. CPU covers the control flow; the GPU case
adds a background load on the default stream and checks the reply is unchanged."""

import os
import threading
import time

import pytest
import torch

from mote.config import Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM
from mote.serve.engine import Engine, GenParams

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _tiny_run(tmp_path):
    cfg = MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    run = tmp_path / "runs" / "pilot_tiny"
    run.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": 3, "config": cfg.to_dict(), "extra": {"bytes_seen": 3000}}, run / "last.pt")
    return run


def _engine(tmp_path, monkeypatch, gated: bool):
    monkeypatch.setenv("MOTE_SERVE_GATED", "1" if gated else "0")
    run = _tiny_run(tmp_path)
    eng = Engine(run / "last.pt", device=DEV)
    eng.gpu_gate = threading.Lock()
    return eng


def _reply(eng, max_bytes=24):
    out = []
    eng.generate([{"role": "user", "content": "hello there"}], GenParams(max_bytes=max_bytes, temperature=0.0), out.append, threading.Event())
    done = [e for e in out if e.get("type") == "done"]
    assert done, out
    return done[0]["text"]


@pytest.mark.skipif(DEV != "cuda", reason="the GPU gate exists to keep serving off the card while a job "
                                          "trains; on CPU there is nothing to contend for, so it is "
                                          "never held and the assertion below is CUDA-specific")
def test_gate_is_held_only_when_gated(tmp_path, monkeypatch):
    for gated in (False, True):
        eng = _engine(tmp_path / str(gated), monkeypatch, gated)
        seen = {"held": None}
        orig_generate = eng._generate

        def spy(*a, **k):
            seen["held"] = eng.gpu_gate.locked()
            return orig_generate(*a, **k)

        eng._generate = spy
        _reply(eng)
        assert seen["held"] is gated


def test_stream_and_pool_exist_only_on_cuda(tmp_path, monkeypatch):
    eng = _engine(tmp_path, monkeypatch, False)
    if DEV == "cuda":
        assert eng.stream is not None and eng.pool is not None
        assert eng.stream.priority < 0  # high priority
    else:
        assert eng.stream is None and eng.pool is None
    eng.drain()  # no-op on the CPU, a sync on the GPU


def test_reply_identical_with_and_without_gate(tmp_path, monkeypatch):
    a = _reply(_engine(tmp_path / "a", monkeypatch, True))
    b = _reply(_engine(tmp_path / "b", monkeypatch, False))
    assert a == b


@pytest.mark.skipif(DEV != "cuda", reason="needs the GPU")
def test_reply_unchanged_under_a_default_stream_load(tmp_path, monkeypatch):
    eng = _engine(tmp_path, monkeypatch, False)  # the graph decoder runs
    quiet = _reply(eng, 48)
    stop = threading.Event()

    def load():  # a training-like load on the default stream from another thread
        x = torch.randn(2048, 2048, device="cuda")
        while not stop.is_set():
            x = (x @ x).tanh()
        torch.cuda.synchronize()

    t = threading.Thread(target=load)
    t.start()
    try:
        time.sleep(0.2)
        loud = _reply(eng, 48)
    finally:
        stop.set()
        t.join()
    assert loud == quiet
