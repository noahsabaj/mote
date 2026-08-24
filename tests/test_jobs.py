"""The training job queue (mote/serve/jobs.py, docs/shape.md): jobs run to done, cancel stops at a step
boundary with a checkpoint, an interrupted queue auto-resumes on boot, the GPU gate really pauses
training while it is held, the EMA follows the weights, and the serving engine hot-swaps them."""

import json
import threading
import time

import numpy as np
import torch

import mote.serve.app as A
from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.serve.engine import Engine
from mote.serve.jobs import Ema, JobQueue, JobRecord

TINY = dict(d_model_outer=32, encoder_layers=1, decoder_layers=1)


def _cfg():
    return MoteConfig(
        **TINY, main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )


def _fixture(tmp_path):
    _cfg().save(tmp_path / "tiny.json")
    rng = np.random.default_rng(0)
    for split, n in (("train", 20000), ("val", 4000)):
        rng.integers(0, 256, size=n, dtype=np.uint16).tofile(tmp_path / f"tiny.{split}.bin")
    (tmp_path / "tiny.meta.json").write_text(json.dumps({
        "train": {"file": "tiny.train.bin"}, "val": {"file": "tiny.val.bin"}}))
    return tmp_path


def _argv(tmp, out, steps=4):
    return ["--config", str(tmp / "tiny.json"), "--data", str(tmp / "tiny"), "--out", str(out),
            "--batch-size", "2", "--seq-len", "64", "--grad-accum", "2", "--max-steps", str(steps),
            "--eval-every", "1000", "--eval-batches", "1", "--log-every", "1000",
            "--ckpt-minutes", "99999", "--max-minutes", "99999"]


def _wait(pred, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_job_runs_to_done_and_syncs_and_finishes(tmp_path):
    tmp = _fixture(tmp_path)
    gate = threading.Lock()
    synced, finished = [], []
    q = JobQueue(tmp / "jobs.json", gate, sync_steps=2,
                 on_serve_sync=lambda cfg, sd, name, step: synced.append((name, step)),
                 on_finished=lambda rec: finished.append(rec.state))
    q.start()
    rec = q.submit(_argv(tmp, tmp / "runA", steps=5))
    assert _wait(lambda: q.status()["current"] is None and any(r["id"] == rec.id and r["state"] == "done" for r in q.status()["recent"]))
    assert finished == ["done"]
    assert synced and synced[-1][0].endswith("/ema") and synced[-1][1] >= 4  # every 2 steps
    assert (tmp / "runA" / "last.pt").exists()
    q.shutdown()


def test_cancel_running_job_checkpoints_and_moves_on(tmp_path):
    tmp = _fixture(tmp_path)
    q = JobQueue(tmp / "jobs.json", threading.Lock())
    q.start()
    rec = q.submit(_argv(tmp, tmp / "runB", steps=100000))
    assert _wait(lambda: (q.status()["current"] or {}).get("id") == rec.id)
    time.sleep(1.0)
    q.cancel()
    assert _wait(lambda: any(r["id"] == rec.id and r["state"] == "cancelled" for r in q.status()["recent"]))
    assert (tmp / "runB" / "last.pt").exists()  # the graceful stop still saved
    q.shutdown()


def test_interrupted_running_job_reenqueues_with_resume(tmp_path):
    tmp = _fixture(tmp_path)
    state = tmp / "jobs.json"
    argv = _argv(tmp, tmp / "runC")
    state.write_text(json.dumps({"jobs": [{"id": "dead0000", "argv": argv, "state": "running"}]}))
    q = JobQueue(state, threading.Lock())  # no start(): just the boot-time load
    st = q.status()
    assert any(r["id"] == "dead0000" and r["state"] == "interrupted" for r in st["recent"])
    assert len(st["queued"]) == 1 and st["queued"][0]["resumed"] and "--resume" in st["queued"][0]["argv"]


def test_gate_pauses_training_between_slices(tmp_path):
    tmp = _fixture(tmp_path)
    gate = threading.Lock()
    q = JobQueue(tmp / "jobs.json", gate)
    q.start()
    rec = q.submit(_argv(tmp, tmp / "runD", steps=100000))
    assert _wait(lambda: (tmp / "runD" / "log.jsonl").exists())
    with gate:  # a "reply" holds the GPU
        time.sleep(0.5)
        before = (tmp / "runD" / "last.pt").exists(), (tmp / "runD" / "log.jsonl").stat().st_size
        time.sleep(1.5)
        after = (tmp / "runD" / "last.pt").exists(), (tmp / "runD" / "log.jsonl").stat().st_size
        assert before == after  # nothing moved while the gate was held
    q.cancel()
    assert _wait(lambda: q.status()["current"] is None)
    q.shutdown()


def test_ema_math_and_engine_hot_swap(tmp_path):
    m = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        m.weight.fill_(0.0)
    ema = Ema(m, decay=0.5)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m)
    assert torch.allclose(ema.shadow["weight"], torch.full((4, 4), 0.5))
    sd = ema.state_dict(m)
    assert torch.allclose(sd["weight"], torch.full((4, 4), 0.5))

    # engine: EMA weights swap in place, the cache clears, `live` shows the run
    from mote.model.hnet import HNetForCausalLM
    cfg = _cfg()
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    run = tmp_path / "runs" / "seed"
    run.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 1, "config": cfg.to_dict(), "extra": {}}, run / "last.pt")
    eng = Engine(run / "last.pt", device="cpu")
    eng.prefix_cache.put("card", [1, 2, 3], [torch.zeros(8)], torch.zeros(4), 1)
    torch.manual_seed(1)
    other = HNetForCausalLM(cfg)
    eng.apply_run_weights(cfg.to_dict(), other.state_dict(), "runX/ema", 42)
    assert eng.serving_live == "runX/ema@42" and eng.info()["live"] == "runX/ema@42"
    assert len(eng.prefix_cache.items) == 0
    p_eng = dict(eng.model.named_parameters())
    p_oth = dict(other.named_parameters())
    k = next(iter(p_oth))
    assert torch.allclose(p_eng[k], p_oth[k])
