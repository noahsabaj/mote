"""What the tests share: the suite's small model, checkpoints written the way the trainer writes them,
random byte shards in the loader's layout, and the device the trainer may ask for.

Plain functions rather than fixtures on purpose — most tests build their model at module level or
inside a helper of their own, and `from conftest import tiny_cfg` reads better there than a fixture
threaded through three helpers. Before this file, 26 test files carried their own tiny config and 13
wrote their own checkpoint dict by hand; renaming one config field touched twenty files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from mote.config import Mamba3Cfg, MoteConfig, RelationCfg

# The trainer refuses a missing GPU (2026-08-28: nothing falls back to the CPU silently), so a test that
# builds a Trainer asks for the device that is there — a CPU-only run (CUDA_VISIBLE_DEVICES="") runs
# the trainer tests too, only the kernel and graph tests skip.
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tiny_cfg(**overrides) -> MoteConfig:
    """Outer 32, one Mamba-3 layer each side, a one-layer Relation main at 32, a 256-byte window.
    Any `MoteConfig` field overrides (`main=RelationCfg(...)`, `max_seq_len=2048`, `pad_vocab_to=264`)."""
    kw = dict(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )
    kw.update(overrides)
    return MoteConfig(**kw)


def tiny_model(cfg: Optional[MoteConfig] = None, seed: int = 0, **overrides):
    """A freshly initialised model on the given seed (the seed decides the weights, nothing else does)."""
    from mote.model.hnet import HNetForCausalLM

    torch.manual_seed(seed)
    return HNetForCausalLM(cfg if cfg is not None else tiny_cfg(**overrides))


def write_ckpt(path: Path | str, model, step: int = 0, extra: Optional[dict] = None) -> Path:
    """A checkpoint in the trainer's own format (`mote.train.train.save_checkpoint`, no optimizer) —
    what every reader in the repo expects, so a test never fabricates the dict by hand."""
    from mote.train.train import save_checkpoint

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(path, model, None, step, model.cfg, extra or {})
    return path


def write_shard(prefix: Path | str, n_train: int = 20000, n_val: int = 4000, seed: int = 0, sft: bool = False) -> Path:
    """Random byte shards in `mote.data.loader.ByteShard`'s layout: `<prefix>.meta.json` +
    `<prefix>.train.bin` / `.val.bin` (uint16 ids); with `sft=True` the `.sft.` variants plus the
    per-byte loss masks. Returns the prefix the trainer's `--data` takes."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tag = ".sft" if sft else ""
    meta = {}
    for split, n in (("train", n_train), ("val", n_val)):
        f = f"{prefix.name}{tag}.{split}.bin"
        rng.integers(0, 256, size=n, dtype=np.uint16).tofile(prefix.parent / f)
        meta[split] = {"file": f}
        if sft:
            m = f"{prefix.name}{tag}.{split}.mask.bin"
            (rng.random(n) < 0.5).astype(np.uint8).tofile(prefix.parent / m)
            meta[split]["mask_file"] = m
    (prefix.parent / f"{prefix.name}{tag}.meta.json").write_text(json.dumps(meta))
    return prefix


def trainer_argv(cfg_path: Path | str, data_prefix: Path | str, out: Path | str, *extra: str, steps: int = 4) -> list:
    """The trainer argv the tests use: two 64-byte windows a step, no evals, no checkpoint cadence, the
    device that exists, `steps` optimizer steps; `extra` is appended (a later `--device` wins)."""
    return ["--config", str(cfg_path), "--data", str(data_prefix), "--out", str(out), "--device", DEV,
            "--batch-size", "2", "--seq-len", "64", "--grad-accum", "2", "--lr", "1e-3",
            "--max-steps", str(steps), "--eval-every", "1000", "--eval-batches", "1", "--log-every", "1000",
            "--ckpt-minutes", "99999", "--max-minutes", "99999", *extra]
