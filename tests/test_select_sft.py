"""Dynamic SFT selection (signed 2026-08-28, docs/research/curriculum-2026-08-28.md): the trajectory rule that
re-selects windows mid-run, and the trainer plumbing that runs it as a gate-releasing generator."""

import json
from pathlib import Path

import numpy as np
import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.data.select_sft import trajectory_keep, window_starts
from mote.data.loader import ByteShard


# ---- the rule --------------------------------------------------------------------------------
def test_mastered_windows_drop_except_a_floor_share():
    starts = np.arange(10) * 64
    l_now = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, ]) * np.arange(1, 11)  # 0.1 .. 1.0
    keep, rep = trajectory_keep(starts, l_now, None, tau_m=0.35, eps=0.02, floor=0.34, seed=0)
    assert rep["mastered"] == 3 and rep["floor_kept"] == 1  # 0.1, 0.2, 0.3 mastered; a third of them stays
    assert rep["kept"] == 8 and len(keep) == 8 and rep["tau_m"] == 0.35


def test_stuck_windows_drop_and_learnable_ones_stay():
    starts = np.arange(4) * 64
    l_prev = np.array([2.0, 2.0, 2.0, 2.0])
    l_now = np.array([2.0, 1.995, 1.5, 1.0])  # first two moved < eps: stuck; the others learned
    keep, rep = trajectory_keep(starts, l_now, l_prev, tau_m=0.5, eps=0.02, floor=0.0)
    assert rep["stuck"] == 2 and rep["mastered"] == 0 and keep.tolist() == [128, 192]


def test_tau_defaults_to_the_passs_20th_percentile_and_nan_is_unscorable():
    starts = np.arange(6) * 64
    l_now = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0])
    keep, rep = trajectory_keep(starts, l_now, None, tau_m=None, eps=0.02, floor=0.0)
    assert rep["scorable"] == 5 and rep["tau_m"] == np.nanpercentile(l_now, 20)
    assert 0 not in keep.tolist()  # the nan window is never kept


def test_an_empty_keep_falls_back_to_every_scorable_window():
    starts = np.arange(3) * 64
    l_now = np.array([0.1, 0.2, 0.3])
    keep, rep = trajectory_keep(starts, l_now, None, tau_m=1.0, eps=0.02, floor=0.0)
    assert rep["fell_back"] and len(keep) == 3


# ---- the trainer plumbing ---------------------------------------------------------------------
TINY = dict(d_model_outer=32, encoder_layers=1, decoder_layers=1)


def _sft_fixture(tmp_path):
    MoteConfig(**TINY, main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
               mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3),
               mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256).save(tmp_path / "tiny.json")
    rng = np.random.default_rng(0)
    for split, n in (("train", 12000), ("val", 3000)):
        rng.integers(0, 256, size=n, dtype=np.uint16).tofile(tmp_path / f"tiny.sft.{split}.bin")
        (rng.random(n) < 0.5).astype(np.uint8).tofile(tmp_path / f"tiny.sft.{split}.mask.bin")
    (tmp_path / "tiny.sft.meta.json").write_text(json.dumps({
        "train": {"file": "tiny.sft.train.bin", "mask_file": "tiny.sft.train.mask.bin"},
        "val": {"file": "tiny.sft.val.bin", "mask_file": "tiny.sft.val.mask.bin"}}))
    return tmp_path / "tiny"  # ByteShard(prefix, sft=True) reads {prefix}.sft.*


def _argv(tmp_path, prefix, out, extra=()):
    return ["--config", str(tmp_path / "tiny.json"), "--data", str(prefix), "--out", str(out), "--sft", "--no-mbp",
            "--batch-size", "2", "--seq-len", "64", "--grad-accum", "1", "--max-steps", "4", "--eval-every", "1000",
            "--eval-batches", "1", "--log-every", "1", "--ckpt-minutes", "99999", "--max-minutes", "99999",
            "--device", "cpu", *extra]


def test_a_static_keep_restricts_sampling(tmp_path):
    from mote.train.train import Trainer

    prefix = _sft_fixture(tmp_path)
    keep = np.array([0, 640, 1280], dtype=np.int64)
    np.save(tmp_path / "keep.npy", keep)
    t = Trainer(_argv(tmp_path, prefix, tmp_path / "runK", ["--keep", str(tmp_path / "keep.npy")]))
    assert t._main_shard.keep.tolist() == keep.tolist()
    t.close()


def test_reselect_runs_at_the_start_and_at_each_fraction(tmp_path):
    from mote.train.train import Trainer

    prefix = _sft_fixture(tmp_path)
    t = Trainer(_argv(tmp_path, prefix, tmp_path / "runR", ["--reselect-every", "0.5", "--reselect-windows", "8"]))
    assert t._main_shard.keep is None
    for _ in t.run():
        pass
    t.close()
    recs = [json.loads(l) for l in (tmp_path / "runR" / "log.jsonl").read_text().splitlines() if l.strip()]
    passes = [r for r in recs if "reselect" in r]
    assert len(passes) == 2, [r.get("progress") for r in passes]  # pass 0, then at 0.5; none at 1.0
    assert passes[0]["step"] == 0 and passes[0]["reselect"]["scored"] == 8
    assert passes[1]["reselect"]["tau_m"] == passes[0]["reselect"]["tau_m"]  # tau_m fixed by pass 0
    assert isinstance(t._main_shard.keep, np.ndarray) and len(t._main_shard.keep) >= 1


def test_reselect_needs_an_sft_shard(tmp_path):
    import pytest
    from mote.train.train import Trainer

    prefix = _sft_fixture(tmp_path)
    argv = _argv(tmp_path, prefix, tmp_path / "runX", ["--reselect-every", "0.5"])
    argv.remove("--sft")
    with pytest.raises(ValueError, match="needs --sft"):
        Trainer(argv)


def test_window_starts_samples_evenly():
    class S:
        n = 64 * 100 + 1
    assert len(window_starts(S(), 64)) == 100
    assert window_starts(S(), 64, 5) == [0, 64 * 25, 64 * 50, 64 * 74, 64 * 99]
