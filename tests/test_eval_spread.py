"""Val windows: the head of a source-blocked shard is one source; `spread=True` covers the whole shard
(found 2026-08-24 — every trainer val bpb before that read the first source only)."""

import json

import numpy as np
import torch

from mote.data.loader import ByteShard


def _shard(tmp_path, n=100_000):
    # two "sources": the first half is all byte 1, the second half all byte 2 (source-blocked like build_mix)
    arr = np.concatenate([np.full(n // 2, 1, np.uint16), np.full(n - n // 2, 2, np.uint16)])
    arr.tofile(tmp_path / "s.val.bin")
    (tmp_path / "s.meta.json").write_text(json.dumps({"val": {"file": "s.val.bin"}}))
    return ByteShard(tmp_path / "s", "val")


def test_head_reads_one_source_and_spread_reads_both(tmp_path):
    sh = _shard(tmp_path)
    head = torch.cat([ids for ids, _ in sh.sequential_batches(2, 256, max_batches=4)])
    spread = torch.cat([ids for ids, _ in sh.sequential_batches(2, 256, max_batches=4, spread=True)])
    assert head.shape == spread.shape == (8, 257)
    assert set(head.unique().tolist()) == {1}  # the head is the first source only
    assert set(spread.unique().tolist()) == {1, 2}  # spread windows reach the second source
    assert (spread[:4] == 1).all() and (spread[-3:] == 2).all()
    # without a window cap, spread falls back to the full sequential pass
    n_all = sum(1 for _ in sh.sequential_batches(2, 256, spread=True))
    assert n_all == sum(1 for _ in sh.sequential_batches(2, 256))
