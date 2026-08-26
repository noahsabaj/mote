"""Memory-mapped byte shards -> fixed-length training windows (optionally with a loss mask for SFT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from ..tokenizer import (BYTE_VOCAB, CALL_ID, FIM_MIDDLE_ID, FIM_PREFIX_ID, FIM_SUFFIX_ID, OFFSET_ID,
                          R2L_ID, RESULT_ID)


# --- training-time augmentation ---------------------------------------------------------------------
# 2606.16246 ablates three orthogonal categories at 150M in the data-constrained multi-epoch regime and
# all three lower post-decay validation loss (baseline 4.000): random token replacement 3.826, target
# offset prediction 3.870, right-to-left 3.910, and the three combined 3.792. Every one is a transform on
# the byte stream with the model untouched — the same property that makes FIM cheap.
#
# Built 2026-08-26 and OFF by default. The mid-training 2x2 runs without them so its verdict is
# attributable to the data changes alone; these are then their own two-arm comparison, which costs
# nothing extra because the shards are already built and these are flags.

def noise_window(ids: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    """Replace a fraction of real bytes with other random bytes (2606.16246's best single augmentation).

    Only ids below 256 are touched: a special id is structure, not content, and corrupting `<|assistant|>`
    would teach the template wrong. At byte level this is also the typo robustness the spec claims for
    Mote — "a typo is the same word with one byte changed" — which nothing else in the pipeline trains."""
    if rate <= 0:
        return ids
    out = ids.copy()
    content = np.flatnonzero(out < BYTE_VOCAB)
    if not len(content):
        return out
    k = int(round(len(content) * rate))
    if k <= 0:
        return out
    pick = rng.choice(content, size=k, replace=False)
    out[pick] = rng.integers(0, BYTE_VOCAB, size=k)
    return out


def r2l_window(ids: np.ndarray) -> np.ndarray:
    """Reverse the window over CODEPOINTS and mark it with <|r2l|>.

    Reversing raw bytes is not the byte-level analogue of the paper's token reversal — it is corruption:
    b'\xe3\x83\xa9' reversed is not a character at all, and a window of Japanese becomes mojibake. Reversing
    decoded characters preserves the text and, because UTF-8 is a prefix code over the same multiset of
    bytes, preserves the byte length exactly. Special ids are dropped from the reversed span rather than
    scrambled, and the sentinel costs the first position."""
    body = bytes(int(x) for x in ids if x < BYTE_VOCAB)
    try:
        rev = body.decode("utf-8", errors="ignore")[::-1].encode("utf-8")
    except UnicodeError:  # pragma: no cover - decode with errors="ignore" does not raise
        return ids
    out = np.concatenate(([R2L_ID], np.frombuffer(rev, dtype=np.uint8).astype(np.int64)))
    if len(out) >= len(ids):
        return out[: len(ids)]
    return np.concatenate((out, ids[len(out):]))


def offset_window(ids: np.ndarray, i: int) -> np.ndarray:
    """Prepend `<|offset|> <digit>` so the window declares that its target is x_{t+i}, not x_{t+1}.

    Self-describing on purpose: `compute_losses` reads the offset back off the front of the batch, so no
    signature anywhere else has to grow a parameter for a training-only knob. One sentinel plus an ASCII
    digit rather than one id per offset — the embedding has one spare row left, and five offsets would
    need pad_vocab_to raised above 272."""
    if i <= 1:
        return ids
    return np.concatenate(([OFFSET_ID, ord("0") + min(i, 9)], ids[:-2]))


def sample_offset(rng: np.random.Generator, n: int) -> int:
    """Exponentially weighted over {1..n} (2606.16246 Eq. 2, T=1): mass concentrated on i=1 while still
    reaching further, which they describe as an implicit curriculum. Evaluation always uses i=1."""
    if n <= 1:
        return 1
    w = np.exp(-(np.arange(n, dtype=np.float64)))
    return int(rng.choice(np.arange(1, n + 1), p=w / w.sum()))


def fim_window(ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute a window into prefix / suffix / middle around one tool call, in place of left-to-right order.

    2607.12463 mid-trains a coding agent by masking whole *functions* — chosen by dependency analysis, not
    by offset — so the model must reconstruct a call from the text on both sides of it. Mote's tool
    protocol already marks exactly those boundaries: `<|call|> tool: args <|result|>` inside an assistant
    turn. So the cut points are found rather than sampled, and the masked span is always a real call.

    Returns `[FIM_PREFIX] before [FIM_SUFFIX] after [FIM_MIDDLE] call` of the same length, or the window
    unchanged when it contains no complete call (most windows in a general mix do not, which is why this
    is a per-shard mode and not a global one). Three sentinels make the result three ids too long, so the
    *suffix* gives up the difference: the middle is the span being learned and must arrive whole, and a
    truncated one would drop the closing <|result|> that marks where the call ends."""
    calls = np.flatnonzero(ids == CALL_ID)
    if not len(calls):
        return ids
    a = int(rng.choice(calls))
    ends = np.flatnonzero(ids[a:] == RESULT_ID)
    if not len(ends):
        return ids
    b = a + int(ends[0]) + 1  # include the <|result|> that closes the call
    if b >= len(ids) or a == 0:
        return ids
    keep = len(ids) - a - (b - a) - 3  # room left for the suffix once prefix, middle and sentinels are placed
    if keep < 0:
        return ids
    return np.concatenate(([FIM_PREFIX_ID], ids[:a], [FIM_SUFFIX_ID], ids[b : b + keep], [FIM_MIDDLE_ID], ids[a:b]))


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

    def sequential_batches(self, batch_size: int, seq_len: int, max_batches=None, spread: bool = False):
        return self.shards[0].sequential_batches(batch_size, seq_len, max_batches, spread=spread)


class ByteShard:
    """Pretraining shards: ``{prefix}.meta.json`` + ``{prefix}.train.bin`` / ``.val.bin``.
    SFT shards: ``{prefix}.sft.meta.json`` + ``.sft.{split}.bin`` and ``.sft.{split}.mask.bin``."""

    def __init__(self, prefix: str | Path, split: str, sft: bool = False, plain: bool = False,
                 keep: str | Path | None = None, fim: bool = False, seed: int = 0,
                 noise: float = 0.0, r2l: float = 0.0, offset_max: int = 1):
        """`plain`: read an SFT shard's bytes as ordinary LM data (no loss mask) — how a mid-training mix
        takes chat, spec-document and tool bytes (docs/shape.md § mid).

        `keep`: an .npy of window starts from mote.data.select_sft. With it, sample_batch draws only from
        those windows instead of uniformly over the shard — the difficulty selection of 2601.23006 /
        2603.01293, where a smaller, harder SFT set beats the whole of it.

        `fim`: permute each window around one of its tool calls (see `fim_window`). Training only, never
        evaluation — `sequential_batches` is left in reading order so val bpb stays comparable.

        `noise` / `r2l` / `offset_max`: the 2606.16246 augmentations, all training-only for the same
        reason. `noise` is a per-byte replacement rate, `r2l` the share of windows reversed, `offset_max`
        the largest target offset (1 = ordinary next-byte prediction, which is the default everywhere)."""
        prefix = Path(prefix)
        meta_path = prefix.parent / (f"{prefix.name}.sft.meta.json" if sft else f"{prefix.name}.meta.json")
        meta = json.loads(meta_path.read_text())
        self.meta = meta
        self.sft = sft and not plain
        self.data = np.memmap(prefix.parent / meta[split]["file"], dtype=np.uint16, mode="r")
        self.mask = np.memmap(prefix.parent / meta[split]["mask_file"], dtype=np.uint8, mode="r") if self.sft else None
        self.n = len(self.data)
        self.keep = np.load(keep).astype(np.int64) if keep is not None else None
        self.fim = fim
        self.noise, self.r2l, self.offset_max = noise, r2l, max(1, offset_max)
        self._rng = np.random.default_rng(seed)

    def _window(self, s: int, seq_len: int, fim: bool = False, offset: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ids = self.data[s : s + seq_len + 1].astype(np.int64)
        m = self.mask[s : s + seq_len + 1].astype(np.int64) if self.mask is not None else None
        if m is None:  # never augment a masked SFT window: the mask is aligned to the original positions
            if fim:
                ids = fim_window(ids, self._rng)
            # order matters and follows 2606.16246 §2.4: noise corrupts content first, then the sequence
            # is rearranged, then the direction/offset markers are prepended — so the markers themselves
            # are never corrupted.
            if self.noise:
                ids = noise_window(ids, self._rng, self.noise)
            if self.r2l and float(self._rng.random()) < self.r2l:
                ids = r2l_window(ids)
            if offset > 1:
                ids = offset_window(ids, offset)
        return ids, m

    def sample_batch(self, batch_size: int, seq_len: int, generator: torch.Generator):
        """Random windows of seq_len+1 ids (inputs = [:-1], targets = [1:]). Returns (ids, mask|None)."""
        hi = self.n - seq_len - 1
        if self.keep is not None and len(self.keep):
            usable = self.keep[self.keep < hi]
            picks = torch.randint(0, len(usable), (batch_size,), generator=generator).tolist()
            starts = [int(usable[i]) for i in picks]
        else:
            starts = torch.randint(0, hi, (batch_size,), generator=generator).tolist()
        # one offset per micro-batch rather than per sample: the paper draws per sample, but the shard
        # hands back a whole batch and a shared offset keeps the label construction a single shift
        off = sample_offset(self._rng, self.offset_max) if self.offset_max > 1 and self.mask is None else 1
        wins = [self._window(s, seq_len, fim=self.fim, offset=off) for s in starts]
        ids = torch.from_numpy(np.stack([w[0] for w in wins]))
        mask = torch.from_numpy(np.stack([w[1] for w in wins])) if self.mask is not None else None
        return ids, mask

    def sequential_batches(self, batch_size: int, seq_len: int, max_batches: int | None = None, spread: bool = False):
        """Non-overlapping windows for evaluation. Yields (ids, mask|None).

        The val shards are source-blocked (build_mix fills one source after another), so the first
        `max_batches` windows are the first source only. `spread=True` spaces the same number of windows
        evenly over the whole shard instead — every source in proportion (found 2026-08-24; opt-in so
        the lab arms stay comparable to their controls)."""
        n_windows = (self.n - 1) // seq_len
        if spread and max_batches is not None and max_batches * batch_size < n_windows:
            total = max_batches * batch_size
            picks = [round(k * (n_windows - 1) / max(total - 1, 1)) for k in range(total)]
            for b in range(0, total, batch_size):
                wins = [self._window(j * seq_len, seq_len) for j in picks[b : b + batch_size]]
                ids = torch.from_numpy(np.stack([w[0] for w in wins]))
                mask = torch.from_numpy(np.stack([w[1] for w in wins])) if self.mask is not None else None
                yield ids, mask
            return
        for i in range(0, n_windows, batch_size):
            if max_batches is not None and i // batch_size >= max_batches:
                break
            idx = range(i, min(i + batch_size, n_windows))
            wins = [self._window(j * seq_len, seq_len) for j in idx]
            ids = torch.from_numpy(np.stack([w[0] for w in wins]))
            mask = torch.from_numpy(np.stack([w[1] for w in wins])) if self.mask is not None else None
            yield ids, mask
