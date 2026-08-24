"""The Relation decode arena: one preallocated, append-only home for the main network's per-chunk cache.

Decided 2026-08-24 (docs/shape.md, after FreeToken 2608.16157). FullRelation caches {P2, I~} per chunk
(relation.py, Appendix A.5 of the paper) — the only inference state that grows with the context. It used
to live in per-layer tensors rebuilt by `torch.cat` on every forward, and every prefix-cache snapshot copied
all of it (~36 KB a chunk on the flagship, ~108 MB at 3000 chunks). Now every layer's cache is a slice of one
static tensor `[n_layers, 2, H, capacity, dh]`:

* prefill and continuation write their new chunks at rows `[n, n+T)` and read rows `[0, n+T)` as views
  (no copy, no cat; the flash kernel takes the row stride);
* the engine's prefix store keeps only the *Mamba/routing/dechunk* states per anchor (~3 MB) plus the
  arena rows, page by page, on the CPU — an anchor no longer costs a copy of the context;
* the arena is what the decode CUDA graph reads at static addresses, so it stays allocated for the life of
  the engine and its contents stay hot between turns of the same conversation.

Rows at or beyond `n` are scratch: a speculative round or a stopped reply may leave garbage there, and
nothing reads past `n` (the graph masks them out explicitly).
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch


class RelationArena:
    """`buf[layer, 0]` = P2 rows, `buf[layer, 1]` = I~ rows, each [H, capacity, dh]."""

    def __init__(self, n_layers: int, n_heads: int, capacity: int, d_head: int, device, dtype):
        self.n_layers, self.n_heads, self.d_head = n_layers, n_heads, d_head
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.dtype = dtype
        self.buf = torch.zeros(n_layers, 2, n_heads, self.capacity, d_head, device=self.device, dtype=dtype)
        # Which prefix-store branch owns rows [0, n_valid) — the engine's "hot arena" bookkeeping.
        self.owner: Optional[int] = None
        self.n_valid: int = 0
        self.generation: int = 0  # bumps on every reallocation; captured graphs key on it

    @property
    def bytes_per_chunk(self) -> int:
        return self.n_layers * 2 * self.n_heads * self.d_head * self.buf.element_size()

    def nbytes(self) -> int:
        return self.buf.numel() * self.buf.element_size()

    def p2(self, layer: int) -> torch.Tensor:
        return self.buf[layer, 0].unsqueeze(0)  # [1, H, capacity, dh]

    def info(self, layer: int) -> torch.Tensor:
        return self.buf[layer, 1].unsqueeze(0)

    def rows(self, c0: int, c1: int) -> torch.Tensor:
        """All layers' rows [c0, c1): [n_layers, 2, H, c1-c0, dh] (a view)."""
        return self.buf[:, :, :, c0:c1]

    def ensure(self, n_chunks: int) -> bool:
        """Grow (×2, at least n_chunks) when a context needs more rows than the arena holds. Returns True
        when a reallocation happened — the engine then recaptures its graphs and the CPU pages re-sync."""
        if n_chunks <= self.capacity:
            return False
        new_cap = self.capacity
        while new_cap < n_chunks:
            new_cap *= 2
        old = self.buf
        self.buf = torch.zeros(self.n_layers, 2, self.n_heads, new_cap, self.d_head, device=self.device, dtype=self.dtype)
        self.buf[:, :, :, : self.capacity].copy_(old)
        self.capacity = new_cap
        self.generation += 1
        return True

    def invalidate(self) -> None:
        self.owner, self.n_valid = None, 0


class ArenaLayer:
    """What one FullRelation layer sees: its P2/I~ rows and the current fill `n` (shared by all layers)."""

    __slots__ = ("arena", "layer", "state")

    def __init__(self, arena: RelationArena, layer: int, state: "ArenaState"):
        self.arena, self.layer, self.state = arena, layer, state

    @property
    def n(self) -> int:
        return self.state.n

    def write(self, p2: torch.Tensor, info: torch.Tensor) -> None:
        """Append T new chunks at rows [n, n+T) — the caller advances `n` once all layers wrote."""
        T = p2.shape[2]
        n = self.state.n
        assert n + T <= self.arena.capacity, f"arena overflow: {n + T} > {self.arena.capacity}"
        self.arena.buf[self.layer, 0, :, n : n + T].copy_(p2[0])
        self.arena.buf[self.layer, 1, :, n : n + T].copy_(info[0])

    def views(self, upto: int):
        """(P2, I~) over rows [0, upto) as [1, H, upto, dh] views."""
        return self.arena.p2(self.layer)[:, :, :upto], self.arena.info(self.layer)[:, :, :upto]


class ArenaState:
    """The main-network half of an InferenceState: a reference to the shared arena and the chunk count
    `n` this sequence has written. Copying it copies `n` only — the arena is never duplicated."""

    __slots__ = ("arena", "n")

    def __init__(self, arena: RelationArena, n: int = 0):
        self.arena, self.n = arena, int(n)

    def __len__(self) -> int:
        return self.arena.n_layers

    def __getitem__(self, layer: int) -> ArenaLayer:
        return ArenaLayer(self.arena, layer, self)

    def __iter__(self) -> Iterator[ArenaLayer]:
        return (self[i] for i in range(len(self)))

    def copy(self) -> "ArenaState":
        return ArenaState(self.arena, self.n)

    def advance(self, T: int) -> None:
        self.n += int(T)
