"""Latent feedback (2608.08888; docs/results/2026-08-28-latent-feedback-prereg.md): the fusion, the
feedback pass through the H-Net at both levels, the multi-pass objective, Soft decoding, evaluation with
fused prefill passes, and the trainer end to end. CPU only."""

import json

import numpy as np
import pytest
import torch

from mote.config import FeedbackCfg, MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.feedback import FeedbackInput, LatentFusion, feedback_from, plain_mask, shift_right
from mote.model.hnet import HNetForCausalLM
from mote.train.train import Trainer, compute_losses, evaluate, iter_pass_losses, load_checkpoint


def _cfg(level: str, mbp: bool = False) -> MoteConfig:
    return MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=2, d_model=48, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3, enabled=mbp),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
        feedback=FeedbackCfg(level=level),
    )


def _ids(B=2, L=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 250, (B, L), generator=g)


def test_config_round_trip_defaults_to_off():
    d = MoteConfig().to_dict()
    assert d["feedback"] == {"level": "off", "jitter": 0.02}
    d.pop("feedback")  # a checkpoint that predates the field
    assert MoteConfig.from_dict(d).feedback.level == "off"
    assert MoteConfig.from_dict({"feedback": {"level": "chunk"}}).feedback.level == "chunk"


def test_fusion_shapes_and_shift():
    f = LatentFusion(8, 6)
    u = f(torch.randn(2, 5, 8), torch.randn(2, 5, 6))
    assert u.shape == (2, 5, 6)
    h = torch.arange(6.0).view(1, 3, 2)
    s = shift_right(h)
    assert torch.equal(s[0, 0], torch.zeros(2)) and torch.equal(s[0, 1:], h[0, :-1])
    m = plain_mask(2, 5, torch.tensor([0, 2]), "cpu")
    assert m.tolist() == [[True, False, False, False, False], [True, True, True, False, False]]


@pytest.mark.parametrize("level", ["byte", "chunk"])
def test_all_plain_feedback_pass_equals_the_plain_forward(level):
    torch.manual_seed(0)
    model = HNetForCausalLM(_cfg(level)).eval()
    ids = _ids()
    with torch.no_grad():
        out = model(ids)
        assert out.top is not None and (out.front is not None) == (level == "chunk")
        T = out.top.shape[1]
        allplain = FeedbackInput(top=out.top, plain=torch.ones(ids.shape[0], T, dtype=torch.bool), front=out.front)
        again = model(ids, feedback=allplain)
        fused = model(ids, feedback=feedback_from(out, None, detach=True, mixin=False))
    assert torch.equal(again.logits, out.logits)  # every position plain: nothing changes
    assert not torch.allclose(fused.logits, out.logits)  # fused positions change the prediction


@pytest.mark.parametrize("level", ["byte", "chunk"])
def test_multi_pass_loss_trains_the_fusion(level):
    torch.manual_seed(0)
    model = HNetForCausalLM(_cfg(level)).train()
    batch = _ids(L=41)
    gen = torch.Generator().manual_seed(1)
    loss, n, stats, out = compute_losses(model, batch, 5.0, 0.0, 0.03, passes=3, gen=gen)
    assert "ce_fb_sum" in stats and float(n) == batch.shape[0] * (batch.shape[1] - 1)
    loss.backward()
    assert model.fusion.w_u.weight.grad is not None and model.fusion.w_u.weight.grad.abs().sum() > 0
    # the same mixture drawn on the same generator is the same loss: multi-pass training stays reproducible
    model.zero_grad(set_to_none=True)
    loss2, *_ = compute_losses(model, batch, 5.0, 0.0, 0.03, passes=3, gen=torch.Generator().manual_seed(1))
    assert torch.equal(loss.detach(), loss2.detach())
    # detach mode: per-pass backward works and reaches the fusion through the later passes
    model.zero_grad(set_to_none=True)
    for j, (lk, nk, sk, _o) in enumerate(iter_pass_losses(model, batch, 5.0, 0.0, 0.03, passes=2, gen=gen, detach=True)):
        (lk if j == 0 else lk / 1).backward()
    assert model.fusion.w_g.weight.grad is not None


@pytest.mark.parametrize("level", ["byte", "chunk"])
def test_soft_decoding_carries_the_top_state(level):
    torch.manual_seed(0)
    model = HNetForCausalLM(_cfg(level)).eval()
    ids = _ids(B=1, L=24)
    state = model.allocate_inference_state("cpu")
    with torch.no_grad():
        model.prefill(ids, state)
        carried = state.h_prev if level == "byte" else state.z_prev
        assert carried is not None and carried.shape[1] == 1
        before = carried.clone()
        for b in (65, 66, 67):
            logits, routing, is_b, _ = model.step(torch.tensor([[b]]), state)
        after = state.h_prev if level == "byte" else state.z_prev
        assert after is not None and after.shape == before.shape
        if level == "byte":
            assert not torch.equal(after, before)  # updated every byte
        moved = model.move_state(state, "cpu")
        assert torch.equal((moved.h_prev if level == "byte" else moved.z_prev), after)
        # a plain model of the same shape decodes the same prefix differently: the fusion is live
        plain = HNetForCausalLM(_cfg("off")).eval()
        sd = {k: v for k, v in model.state_dict().items() if not k.startswith("fusion.")}
        plain.load_state_dict(sd)
        st2 = plain.allocate_inference_state("cpu")
        plain.prefill(ids, st2)
        l2, *_ = plain.step(torch.tensor([[65]]), st2)
        st3 = model.allocate_inference_state("cpu")
        model.prefill(ids, st3)
        l3, *_ = model.step(torch.tensor([[65]]), st3)
        assert not torch.allclose(l2, l3)


def _shard(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 20000), ("val", 4000)):
        rng.integers(0, 256, size=n, dtype=np.uint16).tofile(tmp_path / f"tiny.{split}.bin")
    (tmp_path / "tiny.meta.json").write_text(json.dumps({"train": {"file": "tiny.train.bin"}, "val": {"file": "tiny.val.bin"}}))
    return tmp_path / "tiny"


def test_evaluate_reports_fused_prefill_passes(tmp_path):
    from mote.data.loader import ByteShard
    torch.manual_seed(0)
    model = HNetForCausalLM(_cfg("chunk")).eval()
    shard = ByteShard(_shard(tmp_path), "val")
    res = evaluate(model, shard, 2, 64, 1, torch.device("cpu"), 5.0, feedback_passes=2)
    assert {"val_bpb", "val_bpb_fb1", "val_bpb_fb2"} <= set(res)
    plain = evaluate(HNetForCausalLM(_cfg("off")), shard, 2, 64, 1, torch.device("cpu"), 5.0, feedback_passes=2)
    assert "val_bpb_fb1" not in plain  # nothing to fuse in a plain model


def _argv(cfg_path, prefix, out, extra=()):
    return ["--config", str(cfg_path), "--data", str(prefix), "--out", str(out), "--device", "cpu",
            "--batch-size", "2", "--seq-len", "64", "--grad-accum", "2", "--lr", "1e-3",
            "--eval-every", "1000", "--eval-batches", "1", "--log-every", "1",
            "--ckpt-minutes", "99999", "--max-minutes", "99999", *extra]


def test_trainer_runs_feedback_arms_and_inits_from_a_plain_checkpoint(tmp_path):
    prefix = _shard(tmp_path)
    plain_cfg = tmp_path / "plain.json"
    _cfg("off").save(plain_cfg)
    t = Trainer(_argv(plain_cfg, prefix, tmp_path / "plain", ["--max-steps", "2"]))
    for _ in t.run():
        pass
    t.close()
    # a fresh feedback run initialised from the plain checkpoint (the arms start from the trunk snapshot)
    for level, extra in (("chunk", []), ("byte", ["--feedback-detach", "--feedback-window", "32"])):
        out = tmp_path / f"fb_{level}"
        t = Trainer(_argv(plain_cfg, prefix, out, ["--max-steps", "3", "--feedback", level, "--feedback-mix", "0,0.5,0.5",
                                                   "--eval-feedback-passes", "1", *extra]))
        assert t.cfg.feedback.level == level and t.model.fusion is not None
        for _ in t.run():
            pass
        t.close()
        lines = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
        steps = [l for l in lines if "ce_fb" in l]
        assert steps and all(2.0 <= l["passes"] <= 3.0 for l in steps)
        ev = [l["eval"] for l in lines if "eval" in l][-1]
        assert "val_bpb_fb1" in ev
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        assert ck["config"]["feedback"]["level"] == level and any(k.startswith("fusion.") for k in ck["model"])
    # --init-from a plain checkpoint into a feedback config loads everything but the fresh fusion
    m = HNetForCausalLM(_cfg("byte"))
    step, _ = load_checkpoint(tmp_path / "plain" / "last.pt", m, None, allow_new=("fusion.",))
    assert step == 2
    with pytest.raises(RuntimeError):
        load_checkpoint(tmp_path / "plain" / "last.pt", HNetForCausalLM(_cfg("byte")), None)  # strict without allow_new
