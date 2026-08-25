"""Serving-beside-training root (signed 2026-08-24 night): a released engine holds no arena / graphs / pool,
serves the same bytes through a per-reply arena, keeps its prefix store working, and re-arms; the job queue
fires the release / idle hooks around jobs; padding rows of the vocabulary are never produced."""

import threading

import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM
from mote.serve.engine import Engine, GenParams, _sample
from mote.tokenizer import END_THINK_ID, THINK_ID, VOCAB_SIZE, ByteTokenizer


def _cfg(**kw):
    return MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3, enabled=False),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256, **kw,
    )


def _engine(tmp_path, **kw):
    cfg = _cfg()
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    run = tmp_path / "runs" / "tiny"
    run.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": 3, "config": cfg.to_dict(), "extra": {}}, run / "last.pt")
    return Engine(run / "last.pt", device="cpu", **kw)


def _reply(eng, text, script):
    ev = []
    eng.generate([{"role": "user", "content": text}], GenParams(max_bytes=len(script), script=list(script), n_candidates=0),
                 ev.append, threading.Event())
    done = ev[-1]
    assert done["type"] == "done"
    return [e for e in ev if e["type"] == "byte"], done


def test_release_serves_the_same_bytes_and_keeps_the_prefix_store(tmp_path):
    eng = _engine(tmp_path)
    assert not eng.released and eng.arena is not None and eng.info()["arena"]["released"] is False
    script = list(b"hello there")
    warm, _ = _reply(eng, "one", script)
    rep = eng.release()
    assert rep["released"] and eng.released and eng.arena is None and eng._gd is None and eng.pool is None
    assert eng.info()["arena"] == {"chunks": 0, "bytes": 0, "hot_branch": None, "hot_chunks": 0, "graph_decode": False, "released": True}
    cold, done = _reply(eng, "one", script)  # the same prompt: the store's anchor rehydrates a per-reply arena
    assert [b["byte"] for b in cold] == [b["byte"] for b in warm]
    assert all(abs(a["p"] - b["p"]) < 1e-4 for a, b in zip(cold, warm))
    assert done["stats"]["bytes"] == len(script)
    assert eng.prefix_cache.report()["branches"] >= 1
    assert eng.rewarm() == {"branches": 0, "bytes": 0, "ms": 0.0}  # nothing resident to warm while released
    assert eng.warmup() == 0.0
    assert eng.rearm() >= 0.0 and not eng.released and eng.arena is not None
    again, _ = _reply(eng, "one", script)
    assert [b["byte"] for b in again] == [b["byte"] for b in warm]
    assert eng.rearm() == 0.0  # idempotent


def test_released_replies_hold_the_gpu_gate(tmp_path, monkeypatch):
    import contextlib

    eng = _engine(tmp_path)
    assert isinstance(eng._reply_gate(), contextlib.nullcontext)  # no gate set: nothing to hold
    gate = threading.Lock()
    eng.gpu_gate = gate
    assert isinstance(eng._reply_gate(), contextlib.nullcontext)  # warm + ungated: concurrent decode (signed design)
    eng.release()
    assert eng._reply_gate() is gate  # released: the whole reply pauses training slices
    monkeypatch.setenv("MOTE_SERVE_RELEASED_GATED", "0")
    assert isinstance(eng._reply_gate(), contextlib.nullcontext)
    monkeypatch.delenv("MOTE_SERVE_RELEASED_GATED")
    seen = {}

    def probe():
        seen["free"] = gate.acquire(timeout=0.05)  # False while the reply holds it
        if seen["free"]:
            gate.release()

    script = list(b"abcdefgh")
    ev = []
    t = None

    def emit(e):
        nonlocal t
        ev.append(e)
        if e["type"] == "byte" and e["i"] == 1 and t is None:
            t = threading.Thread(target=probe)
            t.start()
            t.join()  # probe while the reply is in flight (the reply thread waits for the 50 ms timeout)
    eng.generate([{"role": "user", "content": "gate"}], GenParams(max_bytes=len(script), script=script, n_candidates=0), emit, threading.Event())
    assert seen["free"] is False and ev[-1]["type"] == "done"
    assert gate.acquire(timeout=1.0)  # released after the reply
    gate.release()


def test_triton_autotuner_lock_installs_once():
    from mote.model import triton_lock

    ok = triton_lock.install()
    try:
        from triton.runtime.autotuner import Autotuner
    except Exception:
        assert ok is False
        return
    assert ok and Autotuner._mote_locked and hasattr(Autotuner.run, "_mote_orig")
    first = Autotuner.run
    assert triton_lock.install() and Autotuner.run is first  # idempotent: not wrapped twice


def test_engine_can_start_released(tmp_path):
    eng = _engine(tmp_path, released=True)
    assert eng.released and eng.arena is None
    bytes_, done = _reply(eng, "two", list(b"abc"))
    assert [b["byte"] for b in bytes_] == list(b"abc") and done["reason"] == "max_bytes"


def test_queue_hooks_fire_around_jobs(tmp_path, monkeypatch):
    try:
        from tests.test_jobs import _FakeTrainer, _argv, _fake_queue, _wait
    except ImportError:  # pytest's rootdir import mode
        from test_jobs import _FakeTrainer, _argv, _fake_queue, _wait

    q = _fake_queue(tmp_path, monkeypatch)
    events = []
    q.on_started = lambda rec: events.append(("started", rec.argv[rec.argv.index("--out") + 1]))
    q.on_idle = lambda: events.append(("idle",))
    _FakeTrainer.plans[str(tmp_path / "runA")] = ["slow"]
    q.start()
    assert _wait(lambda: ("idle",) in events)  # an empty queue is idle once
    q.submit(_argv(tmp_path, tmp_path / "runA"))
    q.submit(_argv(tmp_path, tmp_path / "runB"))
    assert _wait(lambda: sum(r.state == "done" for r in q.jobs) == 2 and events[-1] == ("idle",))
    names = [e for e in events if e[0] == "started"]
    assert [n[1] for n in names] == [str(tmp_path / "runA"), str(tmp_path / "runB")]
    assert events.count(("idle",)) == 2  # once before the jobs, once after — not between them
    q.shutdown()


def test_padding_rows_are_masked_and_never_sampled():
    cfg = _cfg(pad_vocab_to=272)
    assert cfg.vocab_size == VOCAB_SIZE == 266 and THINK_ID == 264 and END_THINK_ID == 265
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg).eval()
    ids = torch.randint(0, 256, (2, 40))
    out = model(ids)
    assert out.logits.shape[-1] == 272
    assert torch.isinf(out.logits[..., VOCAB_SIZE:]).all() and torch.isfinite(out.logits[..., :VOCAB_SIZE]).all()
    lg = out.logits[0, -1]
    seen = {_sample(lg, 4.0, 1.0)[0] for _ in range(300)}
    assert max(seen) < VOCAB_SIZE
    # the decode path masks too, and the tokenizer names the new ids
    state = model.allocate_inference_state("cpu")
    model.prefill(ids[:1], state)
    lg1, *_ = model.step(torch.tensor([[65]]), state)
    assert torch.isinf(lg1[0, -1, VOCAB_SIZE:]).all()
    assert ByteTokenizer().decode([THINK_ID, 65, END_THINK_ID], skip_special_tokens=False) == "<|think|>A<|end_think|>"
    # an old 264-row checkpoint keeps its shape through its own config
    old = HNetForCausalLM(_cfg(pad_vocab_to=264))
    assert old.lm_head.weight.shape[0] == 264 and torch.isfinite(old(ids).logits).all()
