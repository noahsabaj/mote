"""Prefix-state cache for the serving engine (decided 2026-08-23, docs/context.md).

Snapshots of the model's inference state after byte sequences it has already read, kept in CPU memory
under a byte budget. A new prompt reuses the longest snapshot whose consumed bytes are a prefix of the
prompt, and only the remainder is read (`HNetForCausalLM.forward_from_state`). Three kinds of snapshot:
``card`` (after the identity card — shared by every conversation), ``prompt`` (end of a turn's prompt)
and ``reply`` (end of the generated reply). Keyed by bytes only: the router is causal, so an identical
byte prefix gives identical chunks and an identical state, up to float rounding (the studio's
"verify prefix cache" toggle measures that rounding; see Engine._verify_prefix).
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch


def _pack(ids: Sequence[int]) -> bytes:
    return array("H", ids).tobytes()  # byte ids are < 512; two bytes each, so prefix checks align


def state_nbytes(o: Any) -> int:
    if isinstance(o, torch.Tensor):
        return o.numel() * o.element_size()
    if isinstance(o, (list, tuple)):
        return sum(state_nbytes(x) for x in o)
    if hasattr(o, "__dataclass_fields__"):
        return sum(state_nbytes(getattr(o, k)) for k in o.__dataclass_fields__)
    return 0


@dataclass
class Snapshot:
    kind: str                 # card | prompt | reply
    ids: bytes                # the consumed byte ids (packed)
    n_ids: int
    state: Any                # InferenceState on the CPU
    logits: torch.Tensor      # [V] next-byte logits after the last consumed byte (CPU)
    n_chunks: int             # chunks closed so far (engine bookkeeping)
    nbytes: int


class PrefixCache:
    """Most-recently-used first; evicts from the tail once the byte budget is exceeded. budget 0 disables."""

    def __init__(self, budget_bytes: int = 1 << 30):
        self.budget = max(int(budget_bytes), 0)
        self.items: List[Snapshot] = []
        self.used = 0
        self.hits = 0
        self.misses = 0

    def peek(self, ids: Sequence[int]) -> Optional[Snapshot]:
        """The longest snapshot whose bytes are a prefix of `ids` (no bookkeeping)."""
        key = _pack(ids)
        best = None
        for s in self.items:
            if s.n_ids <= len(ids) and key.startswith(s.ids) and (best is None or s.n_ids > best.n_ids):
                best = s
        return best

    def lookup(self, ids: Sequence[int]) -> Optional[Snapshot]:
        best = self.peek(ids)
        if best is None:
            self.misses += 1
            return None
        self.hits += 1
        self.items.remove(best)
        self.items.insert(0, best)
        return best

    def put(self, kind: str, ids: Sequence[int], state: Any, logits: torch.Tensor, n_chunks: int) -> Optional[Snapshot]:
        if self.budget == 0:
            return None
        key = _pack(ids)
        for s in [s for s in self.items if s.ids == key]:
            self._drop(s)  # same bytes: the newer snapshot replaces the older one
        snap = Snapshot(kind, key, len(ids), state, logits, n_chunks, state_nbytes(state) + logits.numel() * logits.element_size())
        if snap.nbytes > self.budget:
            return None
        self.items.insert(0, snap)
        self.used += snap.nbytes
        while self.used > self.budget and len(self.items) > 1:
            self._drop(self.items[-1])
        return snap

    def _drop(self, s: Snapshot) -> None:
        self.items.remove(s)
        self.used -= s.nbytes

    def clear(self) -> None:
        self.items.clear()
        self.used = 0

    def report(self) -> dict:
        return {"snapshots": len(self.items), "cache_bytes": self.used, "cache_budget": self.budget,
                "hits": self.hits, "misses": self.misses}
