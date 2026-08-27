"""Prefix store for the serving engine (root design 2026-08-24, docs/context.md; v1 was 2026-08-23).

What survives between turns is split in two, the way FreeToken (2608.16157) splits KV pages from
recurrent-state checkpoints:

* **Branches** — one per linear conversation history. A branch owns the bytes it has read and the
  Relation arena rows for them, stored on the CPU in pages of `PAGE` chunks. Full pages are immutable
  and may be shared between branches (a regenerate or an edit forks a branch: it inherits the full
  pages up to the fork point by reference and copies only the partial page).
* **Anchors** — checkpoints of everything that is *not* the arena (Mamba-3 encoder/decoder states,
  routing, dechunk, the next-byte logits, the multi-byte bookkeeping) at semantic positions: after the
  identity card (`card`, shared by every conversation, never evicted), at the end of a turn's prompt
  (`prompt`, a regenerate reads nothing) and at the end of the reply (`reply`, where the next turn
  starts). ~3 MB each on the flagship instead of the ~108 MB a full snapshot cost.

A new prompt restores the longest anchor whose bytes are a prefix of it, hydrates the arena from the
branch's pages unless the arena already holds that branch (the hot case — zero copies), and reads only
the remainder. Budget: `MOTE_PREFIX_CACHE_MB` over unique pages + anchors; eviction drops whole
least-recently-used branches. Keyed by bytes only: the router is causal, so an identical byte prefix
gives identical chunks and an identical state up to float rounding (Engine._verify_prefix measures it).
"""

from __future__ import annotations

import itertools
import time
from array import array
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..model.arena import RelationArena

PAGE = 256  # chunks per CPU page (~9 MB on the flagship)


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
class Anchor:
    kind: str                 # card | prompt | reply | tool
    n_ids: int                # prefix length (in ids) of the branch's bytes this anchor sits at
    n_chunks: int             # arena rows written up to here
    state: Any                # InferenceState on the CPU (arena rows excluded — they live in the pages)
    logits: torch.Tensor      # [V] next-byte logits after the last consumed byte (CPU)
    nbytes: int
    created: float = field(default_factory=time.time)


class Branch:
    _ids = itertools.count(1)

    def __init__(self, pinned: bool = False):
        self.id = next(Branch._ids)
        self.ids: bytes = b""          # packed ids of everything stored on this branch
        self.pages: List[torch.Tensor] = []
        self.n_chunks = 0
        self.anchors: List[Anchor] = []  # sorted by n_ids
        self.pinned = pinned           # the card: never evicted, never extended (conversations fork from it)
        self.last_used = time.time()

    @property
    def n_ids(self) -> int:
        return len(self.ids) // 2

    def anchor_at(self, n_ids: int) -> Optional[Anchor]:
        return next((a for a in self.anchors if a.n_ids == n_ids), None)


@dataclass
class Hit:
    branch: Branch
    anchor: Anchor

    @property
    def n_ids(self) -> int:
        return self.anchor.n_ids


class PrefixStore:
    """Branches + anchors under a byte budget. budget 0 disables everything (lookups miss, commits no-op)."""

    def __init__(self, budget_bytes: int = 1 << 30, pin: bool = False):
        self.budget = max(int(budget_bytes), 0)
        self.pin = pin
        self.branches: List[Branch] = []
        self.hits = 0
        self.misses = 0
        self.rows_copied_in = 0   # arena rows hydrated from pages (0 on a hot continue — a test counts them)
        self.rows_copied_out = 0

    # ---- lookup -----------------------------------------------------------------------
    def peek(self, ids: Sequence[int]) -> Optional[Hit]:
        """The longest anchor whose bytes are a prefix of `ids` (no bookkeeping)."""
        if not self.budget:
            return None
        key = _pack(ids)
        best: Optional[Hit] = None
        for b in self.branches:
            if b.n_ids == 0:
                continue
            for a in b.anchors:
                if a.n_ids <= len(ids) and key.startswith(b.ids[: 2 * a.n_ids]):
                    if best is None or a.n_ids > best.n_ids or (a.n_ids == best.n_ids and b.last_used > best.branch.last_used):
                        best = Hit(b, a)
        return best

    def lookup(self, ids: Sequence[int]) -> Optional[Hit]:
        hit = self.peek(ids)
        if hit is None:
            if self.budget:
                self.misses += 1
            return None
        self.hits += 1
        hit.branch.last_used = time.time()
        return hit

    # ---- arena <-> pages ----------------------------------------------------------------
    def _new_page(self, arena: RelationArena, rows: int = PAGE) -> torch.Tensor:
        shape = (arena.n_layers, 2, arena.n_heads, int(rows), arena.d_head)
        return torch.empty(shape, dtype=arena.dtype, device="cpu", pin_memory=self.pin and arena.device.type == "cuda")

    def _store_rows(self, b: Branch, arena: RelationArena, c0: int, c1: int) -> None:
        """Copy arena rows [c0, c1) into the branch's pages (c0 == b.n_chunks: append).

        Full pages are exactly PAGE rows — that is what makes them shareable between forks by
        reference. The TAIL page is sized to what the branch actually holds and reallocated as it
        grows. Before 2026-08-27 every page was allocated full, so a branch holding one chunk cost a
        whole page: 27 MiB at the flagship in fp32, 256x what it needed, and with a 1 GiB budget that
        capped the store at ~38 conversations however short they were. This memory is page-locked
        (see `_new_page`), which the host allocator caches rather than returning, so over-allocating
        it is not a cost the OS takes back."""
        assert c0 == b.n_chunks, (c0, b.n_chunks)
        c = c0
        while c < c1:
            pi, off = divmod(c, PAGE)
            take = min(PAGE - off, c1 - c)
            need = off + take
            if pi == len(b.pages):
                b.pages.append(self._new_page(arena, need))
            elif b.pages[pi].shape[3] < need:  # tail page outgrown: one bounded copy, at most PAGE rows
                grown = self._new_page(arena, need)
                grown[:, :, :, :off].copy_(b.pages[pi][:, :, :, :off])
                b.pages[pi] = grown
            b.pages[pi][:, :, :, off : off + take].copy_(arena.rows(c, c + take))
            c += take
        b.n_chunks = c1
        self.rows_copied_out += max(c1 - c0, 0)

    def hydrate(self, b: Branch, n_chunks: int, arena: RelationArena) -> None:
        """Make arena rows [0, n_chunks) hold this branch's chunks. Free when the arena is already hot."""
        assert n_chunks <= b.n_chunks, (n_chunks, b.n_chunks)
        if arena.owner == b.id and arena.n_valid >= n_chunks:
            arena.n_valid = n_chunks
            return
        arena.ensure(n_chunks)
        c = 0
        while c < n_chunks:
            pi, off = divmod(c, PAGE)
            take = min(PAGE - off, n_chunks - c)
            arena.rows(c, c + take).copy_(b.pages[pi][:, :, :, off : off + take])
            c += take
        arena.owner, arena.n_valid = b.id, n_chunks
        self.rows_copied_in += n_chunks

    def _fork(self, parent: Branch, at_chunks: int, arena: RelationArena) -> Branch:
        """A new branch sharing the parent's full pages below `at_chunks`; the partial page is copied."""
        nb = Branch()
        full, rem = divmod(at_chunks, PAGE)
        nb.pages = list(parent.pages[:full])
        if rem:
            nb.pages.append(parent.pages[full][:, :, :, :rem].clone())  # only the rows the fork inherits
        nb.n_chunks = at_chunks
        return nb

    # ---- commit --------------------------------------------------------------------------
    def commit(self, branch: Optional[Branch], from_chunks: int, kind: str, ids: Sequence[int],
               state_cpu: Any, logits: torch.Tensor, n_chunks: int, arena: RelationArena) -> Optional[Branch]:
        """Record an anchor after reading `ids` (state_cpu = the CPU copy of everything but the arena).
        `branch` is where the read started (None = cold), `from_chunks` the arena rows that were valid
        for it at that point. Extends the branch when `ids` continues it, forks when they diverge (or
        the branch is pinned), refreshes the anchor when `ids` is already stored. Returns the branch
        the anchor lives on (None when the store is disabled)."""
        if not self.budget:
            return None
        key = _pack(ids)
        n_ids = len(ids)
        if branch is None:
            target = Branch(pinned=(kind == "card"))
            self.branches.insert(0, target)
            self._store_rows(target, arena, 0, n_chunks)
            target.ids = key
        elif branch.ids.startswith(key) and not branch.pinned:
            target = branch  # already stored: refresh the anchor only
        elif key.startswith(branch.ids) and not branch.pinned:
            target = branch
            if n_chunks > target.n_chunks:
                self._store_rows(target, arena, target.n_chunks, n_chunks)
            target.ids = key
        else:
            target = self._fork(branch, from_chunks, arena)
            self.branches.insert(0, target)
            self._store_rows(target, arena, from_chunks, n_chunks)
            target.ids = key
        old = target.anchor_at(n_ids)
        if old is not None:
            target.anchors.remove(old)
        anc = Anchor(kind, n_ids, n_chunks, state_cpu, logits, state_nbytes(state_cpu) + logits.numel() * logits.element_size())
        target.anchors.append(anc)
        target.anchors.sort(key=lambda a: a.n_ids)
        target.last_used = time.time()
        arena.owner, arena.n_valid = target.id, n_chunks
        self._evict()
        return target

    # ---- budget ----------------------------------------------------------------------------
    def used_bytes(self) -> int:
        seen: Dict[int, int] = {}
        anchors = 0
        for b in self.branches:
            for p in b.pages:
                seen[id(p)] = p.numel() * p.element_size()
            anchors += sum(a.nbytes for a in b.anchors)
        return sum(seen.values()) + anchors

    def _exclusive_pages(self, b: Branch) -> int:
        """Page bytes only `b` holds — what dropping it would actually give back."""
        shared = {id(p) for other in self.branches if other is not b for p in other.pages}
        pages = {id(p): p.numel() * p.element_size() for p in b.pages}
        return sum(n for i, n in pages.items() if i not in shared)

    def _evict(self) -> None:
        """Drop branches until the store is under budget, least useful first.

        Order by what a drop FREES, then by recency. `_fork` gives a child the parent's full pages by
        reference, so a branch whose pages are all shared with a live fork returns nothing but its
        anchors — a few MB against the hundreds the pages weigh. Evicting it first was the worst of
        both: the hot branch keeps the memory, the cold branch's cached states are gone anyway, and
        the loop, seeing the budget unmet, went on to take the fork as well. Measured 2026-08-27 on
        two branches sharing every page: both evicted, store emptied. Now a wholly-shared branch is
        the last resort, so eviction spends itself on branches that actually release pages.

        This is still a hard cap: if the only way under budget is to drop everything, it drops
        everything. What changed is which branch goes first."""
        while self.used_bytes() > self.budget:
            victims = [b for b in self.branches if not b.pinned]
            if not victims:
                break
            # (frees no pages, then oldest): branches holding pages of their own go first
            oldest = min(victims, key=lambda b: (self._exclusive_pages(b) <= 0, b.last_used))
            self.branches.remove(oldest)

    def clear(self) -> None:
        self.branches.clear()

    # ---- swap re-warm ---------------------------------------------------------------------
    def rewarm_plan(self, max_age_s: float, max_branches: int) -> List[Tuple[List[int], List[Tuple[str, int]]]]:
        """Which conversations to re-read after a weight swap: (ids, [(kind, n_ids), ...]) for the
        branches used within `max_age_s`, most recent first, the card excluded (it is re-read anyway)."""
        now = time.time()
        recent = sorted((b for b in self.branches if not b.pinned and now - b.last_used <= max_age_s),
                        key=lambda b: -b.last_used)[:max_branches]
        out = []
        for b in recent:
            ids = list(array("H", b.ids))
            out.append((ids, [(a.kind, a.n_ids) for a in b.anchors]))
        return out

    def report(self) -> dict:
        return {
            "snapshots": sum(len(b.anchors) for b in self.branches), "branches": len(self.branches),
            "cache_bytes": self.used_bytes(), "cache_budget": self.budget,
            "hits": self.hits, "misses": self.misses,
        }
