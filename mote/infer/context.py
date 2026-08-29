"""Folding: what Mote sees when a conversation outgrows the window (docs/context.md).

The client keeps the whole conversation; the server decides what fits. When the next prompt would
exceed `limit - reserve`, the oldest turns are *folded* into a compaction card instead of being
dropped: the first user message (what the chat is about), user-stated facts picked by rules, and
then the most recent turns verbatim, as many as fit. The card is merged into the first kept user
turn, so the model sees the user/assistant alternation it was trained on; it is deterministic,
shown to the user, and editable (`card_override`).

No model writes the card. A 35M model cannot read a summary it wrote; the rules are honest about
what they keep and the studio shows the exact bytes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..tokenizer import ByteTokenizer, ChatMessage

FIRST_BYTES = 200      # of the first user message
FACT_BYTES = 80        # per fact sentence
MAX_FACTS = 6
FOLD_SLACK = 0.25  # an auto fold frees this fraction of the window, so the prefix holds for several turns


_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_FACT_START = re.compile(r"^(my |i |i'm |i am |i've |i have |we |our |mine )", re.IGNORECASE)
_NOT_A_FACT = re.compile(r"^(i think|i guess|i wonder|i don't know|i'm not sure|i mean|i see|i need you|i want you|i'd like you)", re.IGNORECASE)


def _clip(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    s = b[:max_bytes].decode("utf-8", errors="ignore")
    cut = s.rfind(" ")
    return (s[:cut] if cut > max_bytes // 2 else s).rstrip() + "…"


def user_facts(messages: Sequence[ChatMessage], max_facts: int = MAX_FACTS) -> List[str]:
    """First-person declarative sentences from the user's turns, in order, deduplicated:
    "My dog's name is Biscuit." yes; "Do you like Lisbon?" no; "I think so." no."""
    out: List[str] = []
    seen = set()
    for m in messages:
        if m.role != "user":
            continue
        for s in _SENTENCE.split(m.content):
            s = " ".join(s.split())
            if not s or s.endswith("?") or not _FACT_START.match(s) or _NOT_A_FACT.match(s):
                continue
            key = s.lower().rstrip(".")
            if key in seen:
                continue
            seen.add(key)
            out.append(_clip(s, FACT_BYTES))
            if len(out) >= max_facts:
                return out
    return out


def build_card(folded: Sequence[ChatMessage], max_facts: int = MAX_FACTS, first_bytes: int = FIRST_BYTES) -> str:
    first = next((m.content for m in folded if m.role == "user"), "")
    first = " ".join(first.split())
    n = len(folded)
    parts = [f"(Earlier in this conversation, {n} turn{'s' if n != 1 else ''} folded."]
    if first:
        parts.append(f' You first said: "{_clip(first, first_bytes)}".')
    facts = [f for f in user_facts(folded, max_facts) if f.lower().rstrip(".") != first.lower().rstrip(".")[: len(f)]]
    if facts:
        parts.append(" Notes: " + " ".join(facts))
    parts.append(")")
    return "".join(parts)


@dataclass
class Fold:
    ids: List[int]
    used: int
    limit: int
    folded_from: Optional[int]      # index into the non-system messages where the verbatim part starts
    card: Optional[str]             # the bytes merged into the first kept user turn
    truncated: bool                 # even folding could not fit: oldest turns were dropped

    def report(self) -> Optional[dict]:
        if self.folded_from is None:
            return None
        return {"from": self.folded_from, "turns": self.folded_from, "card": self.card or ""}


def _merge(card: Optional[str], m: ChatMessage) -> ChatMessage:
    return ChatMessage(m.role, f"{card}\n\n{m.content}") if card else m


def fold(messages: Sequence[dict], limit: int, reserve: int, tok: ByteTokenizer, mode: str = "auto",
         card_override: Optional[str] = None, prev: Optional[dict] = None) -> Fold:
    """mode: 'auto' folds only when the prompt would overflow; 'now' folds everything before the last
    user turn; 'off' is plain truncation (drop oldest), kept for the needle probe's baseline.

    Auto folds fold *in batches*: enough to free FOLD_SLACK of the window, and a client that passes
    `prev` = {"from", "card"} (its last fold) keeps that fold point and card for as long as the prompt
    still fits — so the bytes before the newest turn stay identical from turn to turn, which is what
    the engine's prefix cache reuses (decided 2026-08-23)."""
    msgs = [ChatMessage(m["role"], m["content"]) for m in messages]
    system = [m for m in msgs if m.role == "system"]
    rest = [m for m in msgs if m.role != "system"]
    budget = max(limit - reserve, 8)

    def ids_of(ms: Sequence[ChatMessage]) -> List[int]:
        return tok.format_chat(list(system) + list(ms), add_generation_prompt=True)

    whole = ids_of(rest)
    if mode != "now" and len(whole) <= budget:
        return Fold(whole, len(whole), limit, None, None, False)

    if mode == "auto" and prev and isinstance(prev.get("from"), int):
        k = prev["from"]
        if 1 <= k < len(rest) and rest[k].role == "user":
            card = card_override if card_override is not None else str(prev.get("card") or "")
            ids = ids_of([_merge(card, rest[k])] + rest[k + 1:])
            if len(ids) <= budget:
                return Fold(ids, len(ids), limit, k, card, False)

    if mode == "off":
        kept = list(rest)
        truncated = False
        while len(ids_of(kept)) > budget and len(kept) > 1:
            kept = kept[1:]
            truncated = True
        ids = ids_of(kept)
        return Fold(ids, len(ids), limit, None, None, truncated)

    user_idx = [k for k, m in enumerate(rest) if m.role == "user" and k >= 1]
    if mode == "now":
        user_idx = user_idx[-1:]
    # Fold a little more rather than lose the card: every fold point is tried with the card first.
    # Auto mode first tries to leave FOLD_SLACK of the window free (fold in batches, not one turn a time).
    targets = [max(budget - int(limit * FOLD_SLACK), 8), budget] if mode == "auto" else [budget]
    for target in targets:
        for k in user_idx:
            folded = rest[:k]
            for max_facts in (MAX_FACTS, 3, 0):
                card = card_override if card_override is not None else build_card(folded, max_facts)
                kept = [_merge(card, rest[k])] + rest[k + 1:]
                ids = ids_of(kept)
                if len(ids) <= target:
                    return Fold(ids, len(ids), limit, k, card, False)
                if card_override is not None:
                    break
    # no fold point fits with a card (a single huge recent turn): keep what fits without one
    for k in user_idx:
        ids = ids_of(rest[k:])
        if len(ids) <= budget:
            return Fold(ids, len(ids), limit, k, "", False)

    # nothing fits even with a single turn: plain truncation of a giant message
    kept = rest[-1:] if rest else []
    ids = ids_of(kept)
    return Fold(ids, len(ids), limit, len(rest) - 1 if len(rest) > 1 else None, "", True)


def context_report(messages: Sequence[dict], limit: int, reserve: int, tok: ByteTokenizer, mode: str = "auto",
                   card_override: Optional[str] = None, prev: Optional[dict] = None) -> dict:
    """What the next prompt would look like, for the studio's meter and fold line — no generation."""
    f = fold(messages, limit, reserve, tok, mode, card_override, prev)
    return {"used": f.used, "limit": f.limit, "reserve": reserve, "fold": f.report(), "truncated": f.truncated,
            "ids": f.ids}
