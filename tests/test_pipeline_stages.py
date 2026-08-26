"""The pipeline's stage mechanics (docs/shape.md § pre/mid/post): the schedules, plain-LM and FIM mixes of
SFT shards, fresh-document skipping for mixes B/C, local-JSONL shards, and a trunk -> snapshot -> branch
whose step horizon survives a stop + resume.

Re-signed 2026-08-26 for the mid-training rework: `cooldown` (1-sqrt(t) to 0.1x over the whole branch)
became `branch` (constant to 80 %, then linear to zero), `constant` joined it as the other arm of the
2x2, and the ANNEAL long-document share is asserted rather than assumed."""

import json

import numpy as np
import pytest
import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.data.build_local import build_local
from mote.data.build_mix import skip_after, skip_docs
from mote.data.loader import ByteShard, MixedShard
from mote.data.sources import ANNEAL, FLAGSHIP
from mote.tokenizer import ASSISTANT_ID, BOS_ID, EOS_ID
from mote.train.train import Trainer, parse_mix_spec, schedule_lr, wsd_lr


def test_three_schedules():
    base = 1e-3
    for s in range(0, 1000, 37):
        assert schedule_lr("wsd", s, 1000, base) == wsd_lr(s, 1000, base)
    assert schedule_lr("trunk", 0, 1000, base) == pytest.approx(base / 100)  # warmup ramp
    assert schedule_lr("trunk", 99, 1000, base) == pytest.approx(base)
    assert all(schedule_lr("trunk", s, 1000, base) == base for s in (100, 500, 999, 5000))  # never decays
    # `branch` (re-signed 2026-08-26): constant for 80 %, then straight down. The old `cooldown` decayed
    # over the whole branch as 1-sqrt(t), which was at 55 % of peak by the first quarter — the shape
    # Index-1.9B (2607.09885 §6.5) measured as *worse than not curating at all*, because the model meets
    # the new data mixture with too little learning rate left to adapt to it.
    br = [schedule_lr("branch", s, 1000, base, min_ratio=0.0) for s in range(0, 1001, 25)]
    assert br[0] == base and all(x == base for x in br[:32])  # flat while mix C is being absorbed
    assert all(a >= b for a, b in zip(br, br[1:])) and br[-1] == pytest.approx(0.0)
    assert schedule_lr("branch", 900, 1000, base, min_ratio=0.0) == pytest.approx(0.5 * base)  # linear
    assert schedule_lr("cooldown", 900, 1000, base, min_ratio=0.0) == pytest.approx(0.5 * base)  # alias
    # `constant`: the no-decay arm of the 2x2 — same data, same length, no decay at all.
    assert all(schedule_lr("constant", s, 1000, base) == base for s in (0, 500, 999, 1000))
    # wsd is untouched: every lab arm on record decayed to 0.1x, and --min-lr-ratio defaults per-schedule
    # rather than globally so a new arm stays comparable to its own control.
    assert schedule_lr("wsd", 1000, 1000, base) == pytest.approx(0.1 * base)


def test_parse_mix_spec():
    assert parse_mix_spec("data/sft_identity:0.05") == ("data/sft_identity", 0.05, False, False)
    assert parse_mix_spec("data/sft_local:0.03:plain") == ("data/sft_local", 0.03, True, False)
    # `fim` implies `plain`: the permutation replaces the loss mask's job, so a masked read would be wrong
    assert parse_mix_spec("data/tool_traces:0.03:fim") == ("data/tool_traces", 0.03, True, True)


def _sft_shard(tmp_path, name="chat", n=4000):
    rng = np.random.default_rng(1)
    rng.integers(0, 256, size=n, dtype=np.uint16).tofile(tmp_path / f"{name}.sft.train.bin")
    rng.integers(0, 2, size=n, dtype=np.uint8).tofile(tmp_path / f"{name}.sft.train.mask.bin")
    (tmp_path / f"{name}.sft.meta.json").write_text(json.dumps({"train": {"file": f"{name}.sft.train.bin", "mask_file": f"{name}.sft.train.mask.bin"}}))
    return tmp_path / name


def _plain_shard(tmp_path, name="web", n=4000):
    np.random.default_rng(2).integers(0, 256, size=n, dtype=np.uint16).tofile(tmp_path / f"{name}.train.bin")
    (tmp_path / f"{name}.meta.json").write_text(json.dumps({"train": {"file": f"{name}.train.bin"}}))
    return tmp_path / name


def test_plain_reads_an_sft_shard_without_its_mask(tmp_path):
    sft = _sft_shard(tmp_path)
    gen = torch.Generator().manual_seed(0)
    _, mask = ByteShard(sft, "train", sft=True).sample_batch(2, 16, gen)
    assert mask is not None and mask.shape == (2, 17)
    plain = ByteShard(sft, "train", sft=True, plain=True)
    assert plain.sft is False and plain.mask is None
    ids, mask = plain.sample_batch(2, 16, gen)
    assert mask is None and ids.shape == (2, 17)
    mixed = MixedShard([ByteShard(_plain_shard(tmp_path), "train"), plain], [0.9, 0.1])
    ids, mask = mixed.sample_batch(4, 16, gen)
    assert ids.shape == (4, 17) and mask is None and mixed.sft is False


def test_skip_docs_lands_after_the_earlier_builds_documents(tmp_path):
    docs = [b"a" * 10, b"b" * 20, b"c" * 30]
    it = iter(docs)
    take = lambda: next(it, None)  # noqa: E731
    n, skipped = skip_docs(take, (10 + 2) + (20 + 2))  # exactly what a build of the first two recorded
    assert (n, skipped) == (2, 34) and next(it) == docs[2]
    for i, (a, b) in enumerate(((100, 7), (50, 3))):
        (tmp_path / f"m{i}.json").write_text(json.dumps({"train": {"per_source_bytes": {"x": a, "y": 1}}, "val": {"per_source_bytes": {"x": b}}}))
    assert skip_after([str(tmp_path / "m0.json"), str(tmp_path / "m1.json")]) == {"x": 160, "y": 2}


def test_anneal_registry_is_the_flagship_sources_reweighted():
    assert [s.key for s in ANNEAL] == [s.key for s in FLAGSHIP]
    assert sum(s.share for s in ANNEAL) == pytest.approx(1.0)
    share = {s.key: s.share for s in ANNEAL}
    flag = {s.key: s.share for s in FLAGSHIP}
    assert share["finemath"] > flag["finemath"] and share["synth"] > flag["synth"] and share["fineweb_edu"] < flag["fineweb_edu"]
    assert all(share[k] > 0 for k in flag) and flag["fineweb_edu"] == 0.25  # FLAGSHIP itself untouched
    # The reweighting must not quietly spend the long documents on math. As first written it cut them from
    # 10.0 % to 8.6 %, which is the direction PRISM (2603.17074 §8.1) measured collapsing RULER@128k from
    # 59.09 to 6.46 at larger amplitude — and `needle_auto` was not a gate decider, so it could have
    # shipped. It is a guard now, and this keeps the mixture on the right side of the same line.
    LONG = ("finewiki_long", "gutenberg", "fineweb_long", "code_long")
    assert sum(share[k] for k in LONG) >= sum(flag[k] for k in LONG)


def test_build_local_plain_and_sft(tmp_path):
    text = tmp_path / "t.jsonl"
    text.write_text("\n".join(json.dumps({"text": f"Ivy had {i} apples."}) for i in range(4)) + "\n")
    chat = tmp_path / "c.jsonl"
    convs = [[{"role": "user", "content": f"How many apples? {i}"}, {"role": "assistant", "content": f"{i} apples."}] for i in range(4)]
    chat.write_text("\n".join(json.dumps({"messages": m}) for m in convs) + "\n")

    meta = build_local(tmp_path / "plain", [text], [chat], sft=False, val_frac=0.25)
    assert meta["train"]["docs"] == 6 and meta["val"]["docs"] == 2
    sh = ByteShard(tmp_path / "plain", "train")
    data = np.asarray(sh.data)
    assert (data == BOS_ID).sum() == 6 and (data == EOS_ID).sum() >= 6 and sh.mask is None
    assert ByteShard(tmp_path / "plain", "val").n == meta["val"]["ids"]

    meta = build_local(tmp_path / "sft", [], [chat], sft=True, val_frac=0.25)
    sh = ByteShard(tmp_path / "sft", "train", sft=True)
    data, mask = np.asarray(sh.data), np.asarray(sh.mask)
    assert meta["train"]["convs"] == 3 and len(mask) == len(data)
    # the mask covers exactly the assistant bytes + their EOS: "<i> apples." is 9 bytes + EOS per conversation
    assert mask.sum() == 3 * 10 and all(data[j] == ASSISTANT_ID for j in np.flatnonzero(np.diff(mask) == 1))


def _fixture(tmp_path):
    cfg = MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3, enabled=False),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )
    cfg_path = tmp_path / "tiny.json"
    cfg.save(cfg_path)
    rng = np.random.default_rng(0)
    for split, n in (("train", 20000), ("val", 4000)):
        rng.integers(0, 256, size=n, dtype=np.uint16).tofile(tmp_path / f"tiny.{split}.bin")
    (tmp_path / "tiny.meta.json").write_text(json.dumps({"train": {"file": "tiny.train.bin"}, "val": {"file": "tiny.val.bin"}}))
    return cfg_path, tmp_path / "tiny"


def _argv(cfg_path, prefix, out, extra=()):
    return ["--config", str(cfg_path), "--data", str(prefix), "--out", str(out),
            "--batch-size", "2", "--seq-len", "64", "--grad-accum", "1", "--lr", "1e-3",
            "--eval-every", "1000", "--eval-batches", "1", "--log-every", "1",
            "--ckpt-minutes", "99999", "--max-minutes", "99999", *extra]


def _log(out):
    return [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]


def test_trunk_snapshots_then_a_branch_keeps_its_horizon_across_resume(tmp_path):
    cfg_path, prefix = _fixture(tmp_path)
    trunk = tmp_path / "trunk"
    t = Trainer(_argv(cfg_path, prefix, trunk, ["--max-steps", "4", "--schedule", "trunk", "--snapshot-steps", "2"]))
    for _ in t.run():
        pass
    t.close()
    lrs = [l["lr"] for l in _log(trunk) if "lr" in l]
    assert lrs[0] == pytest.approx(1e-5) and lrs[1:] == [pytest.approx(1e-3)] * 3  # ramp, then constant
    snap = trunk / "snap_00000002.pt"
    assert snap.exists() and (trunk / "snap_00000004.pt").exists()
    ck = torch.load(snap, map_location="cpu", weights_only=False)
    assert ck["step"] == 2 and "optimizer" not in ck and ck["extra"]["schedule"] == "trunk"

    branch = tmp_path / "branch"
    argv = _argv(cfg_path, prefix, branch, ["--max-steps", "20", "--schedule", "branch", "--init-from", str(snap), "--mix", f"{prefix}:0.1"])
    t = Trainer(argv)
    steps = 0
    for ph, _ in t.run():
        if ph == "step":
            steps += 1
            if steps == 6:
                t.request_stop("test")
    t.close()
    assert t.step == 6 and json.loads((branch / "log.jsonl").read_text().splitlines()[-1]).get("done") is True
    t = Trainer(argv + ["--resume"])
    assert t.step == 6 and t.sched_total == 20  # the horizon came back from the checkpoint
    for _ in t.run():
        pass
    t.close()
    assert t.step == 20
    recs = [l for l in _log(branch) if "lr" in l]
    lrs = [l["lr"] for l in recs]
    # One schedule across the stop and resume: flat for the first 80 % of the horizon, then straight down
    # to --min-lr-ratio (0 by default). The stop lands at step 6, inside the constant phase, so this also
    # checks that a resume does not restart the decay clock.
    assert len(lrs) == 20 and all(a >= b for a, b in zip(lrs, lrs[1:]))
    expect = [schedule_lr("branch", int(min(s / 20, 1.0) * 1000), 1000, 1e-3, min_ratio=0.0) for s in range(20)]
    assert lrs == [pytest.approx(e) for e in expect]
    assert lrs[0] == pytest.approx(1e-3) and lrs[-1] < lrs[0]  # no warmup on a branch, and it does decay
    cfg = MoteConfig.load(cfg_path)
    assert all(l["target_ratio"] == cfg.dc.target_ratio_final for l in recs)  # the ramp is over on a branch
