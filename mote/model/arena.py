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

Rows at or beyond `n` are scratch in the sense that their CONTENT is meaningless: a speculative round
or a stopped reply may leave stale values there, and nothing reads past `n` for its answer. They must
still be finite. The decode graph reads a whole bucket width and masks rows past `S` by driving their
softmax weight to zero — but the weights are then multiplied into the rows, and 0 x NaN is NaN. So the
buffer is allocated zeroed and grows zeroed; only *stale* values are allowed past `n`, never garbage.
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
        # `zeros`, and it is load-bearing — see the "scratch" note in the module docstring, which is
        # true of the EAGER path only. The decode graph reads a fixed bucket width and masks rows at
        # or past `S` by setting their softmax weight to zero, but it still multiplies those weights
        # into the rows: 0 x NaN is NaN, so uninitialised garbage past `n` poisons the output. Tried
        # `empty` on 2026-08-27 to save the memset (4 ms at the flagship);
        # tests/test_graph_decode.py::test_graph_stops_exactly_on_stop_id_and_max_bytes caught it.
        self.buf = torch.zeros(n_layers, 2, n_heads, self.capacity, d_head, device=self.device, dtype=dtype)
        # Which prefix-store branch owns rows [0, n_valid) — the engine's "hot arena" bookkeeping.
        self.owner: Optional[int] = None
        self.n_valid: int = 0
        self.generation: int = 0  # bumps on every reallocation; captured graphs key on it
        # The engine's serving MemPool (long-lived allocations only: this buffer and the decode graphs).
        # Growth allocates inside it so a bigger arena never fragments the trainer's pool.
        self.pool = None

    @staticmethod
    def capacity_for(seq_len: int, bpic: float, bucket: int = 256, margin: float = 1.25) -> int:
        """Rows a `seq_len`-byte context needs at a measured compression of `bpic` bytes per chunk.

        `margin` is headroom over the mean: bpic is an average over a validation set and a single
        prompt can chunk finer than it (code, markup, and non-Latin scripts all do). 1.25 covers the
        spread measured on the 35M without paying for a doubling. `bucket` matches the decode
        graph's capture width so a full context does not straddle one more bucket than it needs."""
        need = int(seq_len / max(float(bpic), 1e-6) * float(margin)) + 1
        return max(-(-need // bucket) * bucket, bucket)

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
        when a reallocation happened — the engine then recaptures its graphs and the CPU pages re-sync.

        Growth is expensive twice over: the old and new buffers are live together while the rows are
        copied, and every captured decode graph is dropped and recaptured (~4.5 s for the first). At
        Mote-138M/16384 that measured a 1296 MiB peak on a ONE-chunk overflow (432 old + 864 new).
        Two things keep it off the hot path. `new_arena` sizes capacity from the checkpoint's own
        measured bytes-per-chunk, so a full context fits without ever calling this. And when nothing
        valid is stored — a fresh conversation, or right after `invalidate()` — there is nothing to
        preserve, so the old buffer is released BEFORE the new one is allocated and the peak is the
        new size alone rather than the sum. Zeroed, not `empty` — the decode graph multiplies masked
        (zero) weights into rows past `n`, so garbage there becomes NaN in the output."""
        if n_chunks <= self.capacity:
            return False
        new_cap = self.capacity
        while new_cap < n_chunks:
            new_cap *= 2
        import contextlib

        ctx = torch.cuda.use_mem_pool(self.pool) if (self.pool is not None and self.device.type == "cuda") else contextlib.nullcontext()
        keep = self.n_valid if self.owner is not None else 0
        old = self.buf
        if keep == 0:  # nothing to carry over: hand the old rows back before asking for the new ones
            self.buf = old = None
        with ctx:
            buf = torch.zeros(self.n_layers, 2, self.n_heads, new_cap, self.d_head, device=self.device, dtype=self.dtype)
        if old is not None:
            buf[:, :, :, :keep].copy_(old[:, :, :, :keep])
        else:
            self.owner, self.n_valid = None, 0
        self.buf = buf
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
    `n` this sequence has written. Copying it copies `n` only — the arena is never duplicated.

    A hybrid main (config `pattern`, 2026-08-29) also carries the recurrent state of each Mamba-3 main
    layer in `mamba` (layer -> Mamba3State); those tensors ARE copied and moved with the state, like the
    outer stacks' — they are the constant-size part, the arena rows the growing part."""

    __slots__ = ("arena", "n", "mamba")

    def __init__(self, arena: RelationArena, n: int = 0, mamba: Optional[dict] = None):
        self.arena, self.n = arena, int(n)
        self.mamba = dict(mamba) if mamba else {}

    def __len__(self) -> int:
        return self.arena.n_layers

    def __getitem__(self, layer: int):
        if layer in self.mamba:
            return self.mamba[layer]
        return ArenaLayer(self.arena, layer, self)

    def set(self, layer: int, state) -> None:
        """Write back a Mamba-3 main layer's new recurrent state (Relation layers write into the arena)."""
        if layer in self.mamba:
            self.mamba[layer] = state

    def map(self, fn) -> "ArenaState":
        """A copy with `fn` applied to every Mamba-3 state tensor; the arena reference is shared."""
        from .tree import map_tree

        return ArenaState(self.arena, self.n, {k: map_tree(v, fn) for k, v in self.mamba.items()})

    def __iter__(self) -> Iterator[ArenaLayer]:
        return (self[i] for i in range(len(self)))

    def copy(self) -> "ArenaState":
        return self.map(lambda t: t.clone()) if self.mamba else ArenaState(self.arena, self.n)

    def advance(self, T: int) -> None:
        self.n += int(T)
