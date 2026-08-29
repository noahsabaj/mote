"""The training job queue (mote/serve/jobs.py, docs/shape.md): jobs run to done, cancel stops at a step
boundary with a checkpoint, an interrupted queue auto-resumes on boot, the GPU gate really pauses
training while it is held, the EMA follows the weights, and the serving engine hot-swaps them."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

import mote.serve.app as A
from mote.config import Mamba3Cfg, MoteConfig, RelationCfg
from mote.serve.engine import Engine
from mote.serve.jobs import Ema, JobQueue, JobRecord

DEV = "cuda" if torch.cuda.is_available() else "cpu"  # the trainer refuses a missing GPU; the tests ask for what is there

TINY = dict(d_model_outer=32, encoder_layers=1, decoder_layers=1)


def _cfg():
    return MoteConfig(
        **TINY, main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
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
    return ["--config", str(tmp / "tiny.json"), "--data", str(tmp / "tiny"), "--out", str(out), "--device", DEV,
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
    rec = q.submit(_argv(tmp, tmp / "runA", steps=5), serve=True)  # on the air: its EMA syncs
    assert _wait(lambda: q.status()["current"] is None and any(r["id"] == rec.id and r["state"] == "done" for r in q.status()["recent"]))
    assert finished == ["done"]
    assert synced and synced[-1][0].endswith("/ema") and synced[-1][1] >= 4  # every 2 steps
    assert (tmp / "runA" / "last.pt").exists()
    # a plain job (an arm) never touches what is served
    synced.clear()
    rec2 = q.submit(_argv(tmp, tmp / "runA2", steps=3))
    assert _wait(lambda: any(r["id"] == rec2.id and r["state"] == "done" for r in q.status()["recent"]))
    assert synced == [] and finished == ["done", "done"]
    q.shutdown()


def test_serve_flag_can_be_toggled_and_survives_a_resume_copy(tmp_path, monkeypatch):
    q = _fake_queue(tmp_path, monkeypatch)
    a = tmp_path / "runA"
    _FakeTrainer.plans[str(a)] = ["slow"]
    q.start()
    ra = q.submit(_argv(tmp_path, a), serve=True)
    assert _wait(lambda: _by_id(q)[ra.id].state == "running")
    assert q.current().serve and q.status()["current"]["serve"]
    assert q.set_serve(None, False).id == ra.id and not q.current().serve  # the pick wins
    assert q.set_serve(ra.id, True).serve
    q.shutdown()
    assert q.join(timeout=30.0)
    copy = next(r for r in q.jobs if r.resumed)
    assert copy.serve  # an interrupted job on the air comes back on the air


def test_phase_is_read_from_the_trainer(tmp_path, monkeypatch):
    q = _fake_queue(tmp_path, monkeypatch)
    a = tmp_path / "runA"
    _FakeTrainer.plans[str(a)] = ["slow"]
    q.start()
    assert q.phase() is None and q.status()["phase"] is None
    q.submit(_argv(tmp_path, a))
    assert _wait(lambda: q.phase() == "eval 2/16")
    assert q.status()["phase"] == "eval 2/16"
    q.shutdown()
    q.join(30.0)


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


def test_resume_keeps_the_clock(tmp_path):
    """A resumed run continues its wall clock (max-minutes, wsd progress, elapsed_min) instead of restarting it."""
    tmp = _fixture(tmp_path)
    from mote.train.train import Trainer
    argv = _argv(tmp, tmp / "runE", steps=3)
    t = Trainer(argv)
    for _ in t.run():
        pass
    t.close()
    extra = torch.load(tmp / "runE" / "last.pt", map_location="cpu", weights_only=False)["extra"]
    assert extra["elapsed_sec"] > 0
    t2 = Trainer(argv + ["--resume"])
    assert t2.step == 3
    time.sleep(0.2)
    assert time.time() - t2.t_start >= extra["elapsed_sec"] + 0.2  # the clock carried over and keeps running
    t2.close()
    # a checkpoint from before the clock was saved falls back to the log's last elapsed_min
    from mote.train.train import last_logged_elapsed_sec
    assert abs(last_logged_elapsed_sec(tmp / "runE" / "log.jsonl") - extra["elapsed_sec"]) < 5.0
    assert last_logged_elapsed_sec(tmp / "nope.jsonl") == 0.0


# ---- OOM retries and the front of the queue --------------------------------------------------
class _FakeTrainer:
    """Stands in for Trainer: `plans[out_dir]` lists what each attempt does ("oom" | "ok" | "slow")."""
    attempts: dict = {}
    plans: dict = {}

    def __init__(self, argv):
        self.argv = list(argv)
        self.out_dir = Path(argv[argv.index("--out") + 1])
        self.model = torch.nn.Linear(2, 2)
        self.cfg = _cfg()
        self.step = 0
        self.stopped_reason = None
        self.phase = "eval 2/16"  # what the queue's phase() reports while this runs
        key = str(self.out_dir)
        n = _FakeTrainer.attempts.get(key, 0)
        _FakeTrainer.attempts[key] = n + 1
        plan = _FakeTrainer.plans.get(key, ["ok"])
        self.behaviour = plan[min(n, len(plan) - 1)]

    def run(self):
        if self.behaviour == "oom":
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 72.00 MiB.")
        for _ in range(3):
            if self.behaviour == "slow":
                time.sleep(0.4)
            self.step += 1
            yield ("step", None)

    def request_stop(self, reason="requested"):
        self.stopped_reason = reason

    def close(self):
        pass


def _fake_queue(tmp_path, monkeypatch, delays=(0.3, 0.3, 0.3), retries=3):
    import mote.serve.jobs as J
    monkeypatch.setattr(J, "make_trainer", _FakeTrainer)
    monkeypatch.setattr(J, "OOM_RETRY_DELAYS", delays)
    monkeypatch.setattr(J, "OOM_RETRIES", retries)
    # a retry waits for the card to have room: without CUDA nothing is ever usable, so the queue tests
    # (which are about ordering, delays and budgets) pretend an 8 GB card with nothing on it
    monkeypatch.setattr(J, "gpu_peak_bytes", lambda: 0)
    monkeypatch.setattr(J, "gpu_usable_bytes", lambda: float(8 << 30))
    _FakeTrainer.attempts.clear()
    _FakeTrainer.plans.clear()
    return JobQueue(tmp_path / "jobs.json", threading.Lock())


def _by_id(q):
    return {r.id: r for r in q.jobs}


def test_oom_retry_goes_to_the_front_after_its_delay(tmp_path, monkeypatch):
    q = _fake_queue(tmp_path, monkeypatch)
    a, b = tmp_path / "runA", tmp_path / "runB"
    _FakeTrainer.plans[str(a)] = ["oom", "ok"]  # one transient OOM, then fine
    _FakeTrainer.plans[str(b)] = ["slow"]
    q.start()
    ra = q.submit(_argv(tmp_path, a))
    rb = q.submit(_argv(tmp_path, b))
    assert _wait(lambda: sum(r.state == "done" for r in q.jobs) == 2)
    recs = _by_id(q)
    assert recs[ra.id].state == "failed" and "out of memory" in recs[ra.id].error.lower()
    retry = next(r for r in q.jobs if r.retry_of == ra.id)
    assert retry.state == "done" and retry.retries == 1 and "--resume" in retry.argv and retry.resumed
    ids = [r.id for r in q.jobs]
    assert ids.index(retry.id) < ids.index(rb.id)  # inserted ahead of B in the queue...
    assert retry.started_at >= recs[ra.id].ended_at + 0.3  # ...but only after its delay
    assert retry.started_at >= recs[rb.id].ended_at  # B (already runnable) flowed around the deferred retry
    assert _FakeTrainer.attempts[str(a)] == 2
    q.shutdown()


def test_oom_gives_up_after_the_retry_budget(tmp_path, monkeypatch):
    q = _fake_queue(tmp_path, monkeypatch, delays=(0.1,), retries=2)
    a, b = tmp_path / "runA", tmp_path / "runB"
    _FakeTrainer.plans[str(a)] = ["oom"]  # structural: every attempt dies
    q.start()
    ra = q.submit(_argv(tmp_path, a))
    rb = q.submit(_argv(tmp_path, b))
    assert _wait(lambda: _by_id(q)[rb.id].state == "done" and sum(r.state == "failed" for r in q.jobs) == 3)
    time.sleep(0.5)
    assert sum(r.state == "failed" for r in q.jobs) == 3 and not any(r.state == "queued" for r in q.jobs)
    lineage = [r for r in q.jobs if r.id == ra.id or r.retry_of is not None]
    assert [r.retries for r in lineage] == [0, 1, 2]
    assert _FakeTrainer.attempts[str(a)] == 3
    q.shutdown()


def test_oom_retry_waits_for_gpu_memory(tmp_path, monkeypatch):
    import mote.serve.jobs as J
    q = _fake_queue(tmp_path, monkeypatch, delays=(0.1,))
    monkeypatch.setattr(J, "gpu_peak_bytes", lambda: 6 << 30)  # the failed run peaked at 6 GB
    usable = {"v": 5 << 30}
    monkeypatch.setattr(J, "gpu_usable_bytes", lambda: float(usable["v"]))
    a, b = tmp_path / "runA", tmp_path / "runB"
    _FakeTrainer.plans[str(a)] = ["oom", "ok"]
    q.start()
    ra = q.submit(_argv(tmp_path, a))
    rb = q.submit(_argv(tmp_path, b))
    assert _wait(lambda: _by_id(q)[rb.id].state == "done")
    time.sleep(0.6)
    retry = next(r for r in q.jobs if r.retry_of == ra.id)
    assert retry.state == "queued" and retry.needs_bytes == (6 << 30) + J.OOM_MARGIN  # waiting: 5 GB usable < 6.4 needed
    usable["v"] = 8 << 30  # the desktop gave the memory back
    q._wake.set()
    assert _wait(lambda: _by_id(q)[retry.id].state == "done")
    q.shutdown()


def test_shutdown_interrupts_the_running_job_and_requeues_it_in_front(tmp_path, monkeypatch):
    q = _fake_queue(tmp_path, monkeypatch)
    a, b = tmp_path / "runA", tmp_path / "runB"
    _FakeTrainer.plans[str(a)] = ["slow"]
    q.start()
    ra = q.submit(_argv(tmp_path, a))
    rb = q.submit(_argv(tmp_path, b))
    assert _wait(lambda: _by_id(q)[ra.id].state == "running")
    q.shutdown()
    assert q.join(timeout=30.0)
    recs = _by_id(q)
    assert recs[ra.id].state == "interrupted" and recs[rb.id].state == "queued"
    copy = next(r for r in q.jobs if r.resumed and r.argv[r.argv.index("--out") + 1] == str(a))
    ids = [r.id for r in q.jobs]
    assert "--resume" in copy.argv and ids.index(copy.id) < ids.index(rb.id)


def test_interrupted_job_resumes_in_front_of_the_queue(tmp_path):
    state = tmp_path / "jobs.json"
    argv_dead, argv_other = _argv(tmp_path, tmp_path / "runC"), _argv(tmp_path, tmp_path / "runD")
    state.write_text(json.dumps({"jobs": [{"id": "dead0000", "argv": argv_dead, "state": "running"},
                                          {"id": "othr0000", "argv": argv_other, "state": "queued"}]}))
    q = JobQueue(state, threading.Lock())  # boot-time load only
    queued = q.status()["queued"]
    assert [r["id"] for r in queued][1] == "othr0000" and queued[0]["resumed"] and "--resume" in queued[0]["argv"]


def test_submit_front_jumps_the_queue(tmp_path):
    q = JobQueue(tmp_path / "jobs.json", threading.Lock())  # never started: pure ordering
    first = q.submit(_argv(tmp_path, tmp_path / "run1"))
    second = q.submit(_argv(tmp_path, tmp_path / "run2"), front=True)
    third = q.submit(_argv(tmp_path, tmp_path / "run3"))
    assert [r["id"] for r in q.status()["queued"]] == [second.id, first.id, third.id]
    q.cancel(first.id)
    assert [r["id"] for r in q.status()["queued"]] == [second.id, third.id]


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
    eng.prefix_cache.commit(None, 0, "card", [1, 2, 3], [torch.zeros(8)], torch.zeros(4), 1, eng.arena)
    assert len(eng.prefix_cache.branches) == 1 and eng.arena.owner is not None
    torch.manual_seed(1)
    other = HNetForCausalLM(cfg)
    eng.apply_run_weights(cfg.to_dict(), other.state_dict(), "runX/ema", 42)
    assert eng.serving_live == "runX/ema@42" and eng.info()["live"] == "runX/ema@42"
    assert len(eng.prefix_cache.branches) == 0 and eng.arena.owner is None  # states under old weights are gone
    p_eng = dict(eng.model.named_parameters())
    p_oth = dict(other.named_parameters())
    k = next(iter(p_oth))
    assert torch.allclose(p_eng[k], p_oth[k])


# ---- the boot-time breaker, usable memory, the fit rule, pause (2026-08-28) --------------------
# A kernel update put the nvidia module 57 s after boot; the daemon was up at 13 s, started a flagship job
# on the CPU, and the kernel OOM-killer took the whole process — three times, because the job came straight
# back to the front each time. These pin what the queue does about that on its own.
def _boot(state: Path, jobs: list):
    state.write_text(json.dumps({"jobs": jobs}))
    return JobQueue(state, threading.Lock())  # boot-time load only


def test_a_job_that_died_before_logging_a_step_gets_one_free_retry(tmp_path):
    q = _boot(tmp_path / "jobs.json", [{"id": "dead0000", "argv": _argv(tmp_path, tmp_path / "runC"), "state": "running"}])
    copy = q.status()["queued"][0]
    assert copy["deaths"] == 1 and copy["resumed"] and q.halted is None  # a reboot can land right after a start


def test_a_second_death_before_the_first_step_holds_the_job_and_halts_the_queue(tmp_path):
    q = _boot(tmp_path / "jobs.json", [
        {"id": "dead0000", "argv": _argv(tmp_path, tmp_path / "runC"), "state": "running", "deaths": 1},
        {"id": "othr0000", "argv": _argv(tmp_path, tmp_path / "runD"), "state": "queued"}])
    st = q.status()
    held = next(r for r in st["recent"] if r["state"] == "held")
    assert held["deaths"] == 2 and "before logging a step" in held["error"]
    assert q.halted and q.halted.startswith(held["id"])
    assert q._next_queued()[0] is None  # nothing starts — not even the unrelated job — until a person releases
    q.release()
    assert q._next_queued()[0].id == "othr0000"
    assert not any(r.state == "queued" and r.resumed for r in q.jobs)  # the held one does not come back on its own


def test_progress_since_the_start_resets_the_death_count(tmp_path):
    out = tmp_path / "runC"
    out.mkdir()
    (out / "log.jsonl").write_text(json.dumps({"step": 700, "ce": 1.0}) + "\n")
    q = _boot(tmp_path / "jobs.json", [{"id": "dead0000", "argv": _argv(tmp_path, out), "state": "running",
                                        "deaths": 1, "start_step": 500}])
    copy = q.status()["queued"][0]
    assert copy["deaths"] == 0 and q.halted is None  # it trained past its start: a power cut, not a crash loop


def test_the_start_step_is_recorded_when_a_job_starts(tmp_path, monkeypatch):
    q = _fake_queue(tmp_path, monkeypatch)
    out = tmp_path / "runA"
    out.mkdir()
    (out / "log.jsonl").write_text(json.dumps({"step": 42}) + "\n" + "not json\n")
    q.start()
    r = q.submit(_argv(tmp_path, out))
    assert _wait(lambda: _by_id(q)[r.id].state == "done")
    assert _by_id(q)[r.id].start_step == 42
    q.shutdown()


def test_usable_memory_counts_the_whole_reservation_and_is_zero_without_cuda(monkeypatch):
    import mote.serve.jobs as J
    monkeypatch.setattr(J.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(J.torch.cuda, "mem_get_info", lambda: (4 << 30, 8 << 30))
    monkeypatch.setattr(J.torch.cuda, "memory_reserved", lambda: 2 << 30)
    monkeypatch.setattr(J.torch.cuda, "memory_allocated", lambda: int(1.5 * 2**30))
    assert J.gpu_usable_bytes() == float(6 << 30)  # free + everything of ours: the engine leaves when a job starts
    assert J.gpu_total_bytes() == 8 << 30
    monkeypatch.setattr(J.torch.cuda, "is_available", lambda: False)
    assert J.gpu_usable_bytes() == 0.0 and J.gpu_total_bytes() == 0  # never `inf`: that started jobs with no GPU


def test_an_oom_retry_that_cannot_fit_the_card_is_held_with_a_reason(tmp_path, monkeypatch):
    import mote.serve.jobs as J
    q = _fake_queue(tmp_path, monkeypatch, delays=(0.1,))
    monkeypatch.setattr(J, "gpu_peak_bytes", lambda: 7 << 30)  # peaked at 7 GB...
    monkeypatch.setattr(J, "gpu_total_bytes", lambda: int(7.25 * 2**30))  # ...on a 7.25 GB card: 7 + 0.375 > 7.25
    a, b = tmp_path / "runA", tmp_path / "runB"
    _FakeTrainer.plans[str(a)] = ["oom"]
    q.start()
    ra = q.submit(_argv(tmp_path, a))
    rb = q.submit(_argv(tmp_path, b))
    assert _wait(lambda: _by_id(q)[rb.id].state == "done")
    held = next(r for r in q.jobs if r.retry_of == ra.id)
    assert held.state == "held" and "cannot fit" in held.error and q.halted is None  # visible; the queue flowed on
    assert _FakeTrainer.attempts[str(a)] == 1
    q.shutdown()


def test_status_says_why_a_queued_job_is_waiting(tmp_path, monkeypatch):
    import mote.serve.jobs as J
    monkeypatch.setattr(J, "gpu_usable_bytes", lambda: float(5 << 30))
    q = _boot(tmp_path / "jobs.json", [
        {"id": "mem00000", "argv": _argv(tmp_path, tmp_path / "runA"), "state": "queued", "needs_bytes": 6 << 30},
        {"id": "time0000", "argv": _argv(tmp_path, tmp_path / "runB"), "state": "queued", "not_before": time.time() + 600},
        {"id": "free0000", "argv": _argv(tmp_path, tmp_path / "runC"), "state": "queued"}])
    why = {r["id"]: r["waiting"] for r in q.status()["queued"]}
    assert "6.00 GB" in why["mem00000"] and "5.00 usable" in why["mem00000"]
    assert why["time0000"].startswith("retry in") and why["free0000"] is None


def test_pause_starts_nothing_until_resume(tmp_path, monkeypatch):
    q = _fake_queue(tmp_path, monkeypatch)
    q.pause("waiting for CUDA")
    q.start()
    r = q.submit(_argv(tmp_path, tmp_path / "runA"))
    time.sleep(0.6)
    assert _by_id(q)[r.id].state == "queued" and not q.has_runnable() and q.status()["paused"] == "waiting for CUDA"
    q.resume()
    assert _wait(lambda: _by_id(q)[r.id].state == "done")
    q.shutdown()


def test_the_trainer_refuses_to_fall_back_to_the_cpu(tmp_path, monkeypatch):
    """`--device cuda` is the default and there is no silent fallback: the job that melted the box on
    2026-08-28 would have failed in a millisecond with this message instead of taking 23.5 GB."""
    import pytest
    from mote.train.train import Trainer
    tmp = _fixture(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        Trainer(_argv(tmp, tmp / "runX") + ["--device", "cuda"])
    t = Trainer(_argv(tmp, tmp / "runY") + ["--device", "cpu"])  # asked for on purpose: fine
    assert t.device.type == "cpu"
    t.close()
