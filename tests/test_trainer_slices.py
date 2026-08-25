"""Trainer as a generator (mote/train/train.py, docs/shape.md): draining run() reproduces the old loop,
slices yield at every accumulation micro-batch, request_stop() ends gracefully with a checkpoint, and
--resume continues where a stopped run left off."""

import json

import numpy as np
import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.train.train import Trainer


def _fixture(tmp_path):
    cfg = MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )
    cfg_path = tmp_path / "tiny.json"
    cfg.save(cfg_path)
    rng = np.random.default_rng(0)
    for split, n in (("train", 20000), ("val", 4000)):
        arr = rng.integers(0, 256, size=n, dtype=np.uint16)
        arr.tofile(tmp_path / f"tiny.{split}.bin")
    (tmp_path / "tiny.meta.json").write_text(json.dumps({
        "train": {"file": "tiny.train.bin"}, "val": {"file": "tiny.val.bin"}}))
    return cfg_path, tmp_path / "tiny"


def _argv(cfg_path, prefix, out, extra=()):
    return ["--config", str(cfg_path), "--data", str(prefix), "--out", str(out),
            "--batch-size", "2", "--seq-len", "64", "--grad-accum", "2", "--lr", "1e-3",
            "--eval-every", "1000", "--eval-batches", "1", "--log-every", "1",
            "--ckpt-minutes", "99999", "--max-minutes", "99999", *extra]


def test_drained_run_counts_slices_steps_and_checkpoints(tmp_path):
    cfg_path, prefix = _fixture(tmp_path)
    t = Trainer(_argv(cfg_path, prefix, tmp_path / "run1", ["--max-steps", "4"]))
    assert t.phase == "train"
    phases, seen = [], set()
    for ph, _ in t.run():
        phases.append(ph)
        seen.add(t.phase)  # what the Training sheet shows beside "running" (2026-08-25)
    assert seen == {"train", "eval 1/1"} and t.phase == "train"
    t.close()
    # 4 steps × 2 micro-batches, plus one slice per evaluation window (--eval-batches 1: the final eval) — the
    # daemon slots a reply between eval windows too (2026-08-24)
    assert phases.count("slice") == 4 * 2 + 1 and phases.count("step") == 4
    assert (tmp_path / "run1" / "last.pt").exists()
    lines = [json.loads(l) for l in (tmp_path / "run1" / "log.jsonl").read_text().splitlines()]
    assert lines[-1]["done"] is True and lines[-1]["final_step"] == 4
    assert any("eval" in l and l.get("eval", {}).get("val_bpb") for l in lines)  # the final eval
    ck = torch.load(tmp_path / "run1" / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 4 and "optimizer" in ck


def test_request_stop_then_resume_finishes_the_budget(tmp_path):
    cfg_path, prefix = _fixture(tmp_path)
    out = tmp_path / "run2"
    t = Trainer(_argv(cfg_path, prefix, out, ["--max-steps", "6"]))
    steps = 0
    for ph, _ in t.run():
        if ph == "step":
            steps += 1
            if steps == 2:
                t.request_stop()
    t.close()
    assert torch.load(out / "last.pt", map_location="cpu", weights_only=False)["step"] == 2
    lines = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert any(l.get("stopped") == "requested" for l in lines)

    t2 = Trainer(_argv(cfg_path, prefix, out, ["--max-steps", "6", "--resume"]))
    assert t2.step == 2
    for _ in t2.run():
        pass
    t2.close()
    assert torch.load(out / "last.pt", map_location="cpu", weights_only=False)["step"] == 6


def test_same_seed_same_trajectory(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # GPU atomics (dechunk scatter-add) break bitwise equality
    cfg_path, prefix = _fixture(tmp_path)
    finals = []
    for name in ("a", "b"):
        t = Trainer(_argv(cfg_path, prefix, tmp_path / name, ["--max-steps", "3"]))
        for _ in t.run():
            pass
        t.close()
        lines = [json.loads(l) for l in (tmp_path / name / "log.jsonl").read_text().splitlines()]
        finals.append(next(l["eval"]["val_bpb"] for l in lines if l.get("final")))
    assert finals[0] == finals[1]
