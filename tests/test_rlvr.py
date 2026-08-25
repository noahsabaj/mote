"""RLVR-1 driver (mote/train/rlvr.py): group advantages and the edge-of-competence filter, padded
log-prob gathering, rollouts through the live-model engine with the sim tool and the RL loss mask, one
on-policy update that moves the weights, resume, and the daemon's job dispatch."""

import json

import pytest
import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM
from mote.serve.engine import Engine, GenParams
from mote.serve.jobs import _argparser_for, _job_args, make_trainer
from mote.tokenizer import PAD_ID
from mote.train.rlvr import RlvrTrainer, group_advantages, has_signal, pad_batch, token_logprobs


def _tiny_ckpt(tmp_path):
    cfg = MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3, enabled=False),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=2048,
    )
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    run = tmp_path / "runs" / "init"
    run.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 7, "config": cfg.to_dict(), "extra": {}}, run / "last.pt")
    return run / "last.pt"


def test_advantages_and_signal():
    assert has_signal([1, 0, 0, 1]) and not has_signal([0, 0, 0]) and not has_signal([1.0, 1.0])
    a = group_advantages([1, 0, 0, 1])
    assert a[0] == a[3] == pytest.approx(1.0, abs=1e-4) and a[1] == a[2] == pytest.approx(-1.0, abs=1e-4)
    assert sum(group_advantages([0.2, 0.5, 0.9])) == pytest.approx(0.0, abs=1e-6)


def test_pad_and_logprobs(tmp_path):
    ck = _tiny_ckpt(tmp_path)
    eng = Engine(ck, device="cpu")
    ids, m = pad_batch([[1, 2, 3], [4, 5]], [[0, 1, 1], [0, 1]], torch.device("cpu"))
    assert ids.tolist() == [[1, 2, 3], [4, 5, PAD_ID]] and m.tolist() == [[0, 1, 1], [0, 1, 0]]
    lp = token_logprobs(eng.model, ids, torch.device("cpu"))
    assert lp.shape == (2, 2) and (lp <= 0).all()


def test_engine_from_model_serves_a_live_model(tmp_path):
    ck = _tiny_ckpt(tmp_path)
    ref = Engine(ck, device="cpu")
    live = Engine.from_model(ref.model, ref.cfg, device="cpu", name="rl/policy.pt")
    assert live.ckpt_name == "rl/policy.pt" and live.info()["params"] == ref.info()["params"]
    ev = []
    import threading
    live.generate([{"role": "user", "content": "hi"}], GenParams(temperature=0.0, top_p=1.0, max_bytes=8, n_candidates=0),
                  ev.append, threading.Event(), context={"want_ids": True})
    assert ev[-1]["type"] == "done" and len(ev[-1]["ids"]) == len(ev[-1]["mask"]) and ev[-1]["prompt_ids"]


def _argv(ck, out, extra=()):
    return ["--init-from", str(ck), "--out", str(out), "--steps", "1", "--tasks", "2", "--group", "2", "--max-bytes", "24",
            "--eval-every", "0", "--micro", "2", "--device", "cpu", "--max-minutes", "99", "--ckpt-minutes", "99999", *extra]


def test_one_step_without_signal_then_with_signal(tmp_path, monkeypatch):
    # the tiny random model (272 embedding rows since 2026-08-24) puts <|assistant|> on top and would end every
    # rollout at byte 0; the update needs non-empty rollouts, so only EOS ends one here
    import mote.serve.engine as E
    from mote.tokenizer import EOS_ID
    monkeypatch.setattr(E, "STOP_IDS", {EOS_ID})
    ck = _tiny_ckpt(tmp_path)
    t = RlvrTrainer(_argv(ck, tmp_path / "rl0"))
    phases = [ph for ph, _ in t.run()]
    t.close()
    recs = [json.loads(l) for l in (tmp_path / "rl0" / "log.jsonl").read_text().splitlines()]
    step = next(r for r in recs if "reward" in r)
    assert phases.count("step") == 1 and phases.count("slice") >= 4  # one slice per rollout at least
    assert step["groups"] == 2 and step["success"] == 0.0 and step["groups_used"] == 0 and "note" in step  # a random model never solves a task
    assert (tmp_path / "rl0" / "last.pt").exists() and recs[-1]["done"] is True

    # give the first rollout of every group the reward: the group has signal and the update runs
    t = RlvrTrainer(_argv(ck, tmp_path / "rl1"))
    calls = {"n": 0}

    def fake_reward(env):
        calls["n"] += 1
        return (1.0, 1.0) if calls["n"] % 2 == 1 else (0.0, 0.0)

    t.reward_of = fake_reward
    before = [p.detach().clone() for p in t.model.parameters()]
    for _ in t.run():
        pass
    t.close()
    recs = [json.loads(l) for l in (tmp_path / "rl1" / "log.jsonl").read_text().splitlines()]
    step = next(r for r in recs if "reward" in r)
    assert step["groups_used"] == 2 and "loss" in step and step["kl"] >= 0.0 and step["grad_norm"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, t.model.parameters()))
    # resume continues the task cursor and the step count
    t2 = RlvrTrainer(_argv(ck, tmp_path / "rl1", ["--resume", "--steps", "1"]))
    assert t2.step == 1 and t2.cursor == t.cursor
    t2.close()


def test_daemon_dispatch():
    assert _job_args(["rlvr", "--init-from", "x"]) == ["--init-from", "x"] and _job_args(["--preset", "local"]) == ["--preset", "local"]
    assert _argparser_for(["rlvr"]).prog == "rlvr"
    with pytest.raises(SystemExit):
        _argparser_for(["rlvr"]).parse_args(["--bogus"])
    assert make_trainer.__doc__ and "RlvrTrainer" in make_trainer.__doc__
