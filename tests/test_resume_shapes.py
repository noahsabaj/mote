"""The 264 -> 272 embedding padding (2026-08-24) must not break older runs: a resume rebuilds the model from the
run's saved config, and --init-from an older checkpoint pads its vocabulary rows instead of failing."""

import json

import numpy as np
import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM
from mote.train.train import Trainer, load_checkpoint, pad_vocab_rows


def _cfg(pad):
    return MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1, pad_vocab_to=pad,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3, transition=True),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )


def _data(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 20000), ("val", 4000)):
        rng.integers(0, 256, size=n, dtype=np.uint16).tofile(tmp_path / f"tiny.{split}.bin")
    (tmp_path / "tiny.meta.json").write_text(json.dumps({"train": {"file": "tiny.train.bin"}, "val": {"file": "tiny.val.bin"}}))
    return tmp_path / "tiny"


def _argv(tmp, cfg_path, out, steps, extra=()):
    return ["--config", str(cfg_path), "--data", str(_data(tmp)), "--out", str(out), "--batch-size", "2", "--seq-len", "64",
            "--grad-accum", "1", "--max-steps", str(steps), "--eval-every", "1000", "--eval-batches", "1", "--log-every", "1000",
            "--ckpt-minutes", "99999", "--max-minutes", "99999", *extra]


def test_resume_uses_the_saved_config_not_the_new_default(tmp_path):
    old_cfg, new_cfg = tmp_path / "c264.json", tmp_path / "c272.json"
    _cfg(264).save(old_cfg)
    _cfg(272).save(new_cfg)
    out = tmp_path / "run"
    t = Trainer(_argv(tmp_path, old_cfg, out, 2))
    for _ in t.run():
        pass
    t.close()
    assert t.model.lm_head.weight.shape[0] == 264
    # the "preset default" moved to 272 rows: the resume still rebuilds the run as it was saved
    t2 = Trainer(_argv(tmp_path, new_cfg, out, 3, ("--resume",)))
    assert t2.step == 2 and t2.model.lm_head.weight.shape[0] == 264 and t2.cfg.pad_vocab_to == 264
    t2.close()
    # ...even when a resume that failed under the new default has already rewritten the run's config.json
    # (what happened twice on 2026-08-24): the checkpoint's own config is the authority
    _cfg(272).save(out / "config.json")
    t3 = Trainer(_argv(tmp_path, new_cfg, out, 3, ("--resume",)))
    assert t3.step == 2 and t3.model.lm_head.weight.shape[0] == 264 and t3.cfg.pad_vocab_to == 264
    assert json.loads((out / "config.json").read_text())["pad_vocab_to"] == 264  # and config.json is repaired
    t3.close()


def test_init_from_an_older_checkpoint_pads_the_vocab_rows(tmp_path):
    torch.manual_seed(0)
    old = HNetForCausalLM(_cfg(264))
    ck = tmp_path / "old.pt"
    torch.save({"model": old.state_dict(), "step": 5, "config": old.cfg.to_dict(), "extra": {}}, ck)
    torch.manual_seed(1)
    new = HNetForCausalLM(_cfg(272))
    fresh_tail = new.embeddings.weight[264:].detach().clone()
    step, _ = load_checkpoint(ck, new, None)
    assert step == 5
    assert torch.equal(new.embeddings.weight[:264], old.embeddings.weight)
    assert torch.equal(new.embeddings.weight[264:], fresh_tail)  # spare rows keep the fresh init
    assert torch.equal(new.mbp_head.transition.weight[:264, :264], old.mbp_head.transition.weight)
    # a genuine mismatch is still reported
    sd = old.state_dict()
    sd["embeddings.weight"] = torch.zeros(300, 32)
    padded = pad_vocab_rows(sd, new)
    assert padded["embeddings.weight"].shape[0] == 300  # left alone: rows are never dropped
    try:
        new.load_state_dict(padded)
    except RuntimeError as e:
        assert "size mismatch" in str(e)
    else:
        raise AssertionError("expected a size mismatch")
