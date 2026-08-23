"""Memory-mapped byte shards -> fixed-length training windows (optionally with a loss mask for SFT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


class MixedShard:
    """Several shards sampled per window by weight (e.g. the SFT mix plus a small identity shard)."""

    def __init__(self, shards, weights):
        assert len(shards) == len(weights) and shards
        self.shards = shards
        total = float(sum(weights))
        self.weights = [w / total for w in weights]
        self.sft = shards[0].sft
        self.meta = {"mixed": [getattr(s, "meta", {}) for s in shards], "weights": self.weights}
        self.n = sum(s.n for s in shards)

    def sample_batch(self, batch_size: int, seq_len: int, generator: torch.Generator):
        ids, masks = [], []
        for _ in range(batch_size):
            r = float(torch.rand(1, generator=generator))
            k = 0
            acc = self.weights[0]
            while r > acc and k < len(self.weights) - 1:
                k += 1
                acc += self.weights[k]
            x, m = self.shards[k].sample_batch(1, seq_len, generator)
            ids.append(x)
            masks.append(m)
        return torch.cat(ids, 0), (torch.cat(masks, 0) if masks[0] is not None else None)

    def sequential_batches(self, batch_size: int, seq_len: int, max_batches=None):
        return self.shards[0].sequential_batches(batch_size, seq_len, max_batches)


class ByteShard:
    """Pretraining shards: ``{prefix}.meta.json`` + ``{prefix}.train.bin`` / ``.val.bin``.
    SFT shards: ``{prefix}.sft.meta.json`` + ``.sft.{split}.bin`` and ``.sft.{split}.mask.bin``."""

    def __init__(self, prefix: str | Path, split: str, sft: bool = False):
        prefix = Path(prefix)
        meta_path = prefix.parent / (f"{prefix.name}.sft.meta.json" if sft else f"{prefix.name}.meta.json")
        meta = json.loads(meta_path.read_text())
        self.meta = meta
        self.sft = sft
        self.data = np.memmap(prefix.parent / meta[split]["file"], dtype=np.uint16, mode="r")
        self.mask = np.memmap(prefix.parent / meta[split]["mask_file"], dtype=np.uint8, mode="r") if sft else None
        self.n = len(self.data)

    def _window(self, s: int, seq_len: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ids = self.data[s : s + seq_len + 1].astype(np.int64)
        m = self.mask[s : s + seq_len + 1].astype(np.int64) if self.mask is not None else None
        return ids, m

    def sample_batch(self, batch_size: int, seq_len: int, generator: torch.Generator):
        """Random windows of seq_len+1 ids (inputs = [:-1], targets = [1:]). Returns (ids, mask|None)."""
        hi = self.n - seq_len - 1
        starts = torch.randint(0, hi, (batch_size,), generator=generator).tolist()
        wins = [self._window(s, seq_len) for s in starts]
        ids = torch.from_numpy(np.stack([w[0] for w in wins]))
        mask = torch.from_numpy(np.stack([w[1] for w in wins])) if self.mask is not None else None
        return ids, mask

    def sequential_batches(self, batch_size: int, seq_len: int, max_batches: int | None = None):
        """Non-overlapping windows for evaluation. Yields (ids, mask|None)."""
        n_windows = (self.n - 1) // seq_len
        for i in range(0, n_windows, batch_size):
            if max_batches is not None and i // batch_size >= max_batches:
                break
            idx = range(i, min(i + batch_size, n_windows))
            wins = [self._window(j * seq_len, seq_len) for j in idx]
            ids = torch.from_numpy(np.stack([w[0] for w in wins]))
            mask = torch.from_numpy(np.stack([w[1] for w in wins])) if self.mask is not None else None
            yield ids, mask
