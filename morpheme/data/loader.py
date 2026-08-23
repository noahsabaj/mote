"""Memory-mapped byte shards -> fixed-length training windows (optionally with a loss mask for SFT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


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
