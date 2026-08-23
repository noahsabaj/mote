"""Preference votes: pairs of replies to the same prompt and who preferred which (decided 2026-08-23,
docs/prefs.md). The studio writes pairs and the user's votes; `mote prefs export` hands unrated
pairs to the AI rater (Claude, in session, under docs/rubric.md) and `mote prefs import` takes the
verdicts back. Two append-only JSONL files under data/prefs/ (gitignored with the rest of data/):

    pairs.jsonl   {"id", "ts", "messages": [...context...], "a", "b", "a_source", "b_source", "origin"}
    votes.jsonl   {"pair": id, "ts", "rater": "user" | "claude", "vote": "a" | "b" | "tie" | "both_bad",
                   "reason": "...", "rubric": "<hash or null>"}

A source is {"checkpoint": "overnight_sft/last.pt", "step": 3666, "engine": "current" | "challenger",
"params": {...}}. Votes are never overwritten: the newest vote of a rater for a pair is the one that
counts, so a changed mind after a discussion is a new line, and the history stays.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
PREFS_DIR = ROOT / "data" / "prefs"
PAIRS_FILE = PREFS_DIR / "pairs.jsonl"
VOTES_FILE = PREFS_DIR / "votes.jsonl"
RUBRIC_FILE = ROOT / "docs" / "rubric.md"
VOTES = ("a", "b", "tie", "both_bad")
RATERS = ("user", "claude")

# Phrases the rubric singles out; a pair where only one side has them is worth a rater's time.
RUBRIC_MARKERS = [
    r"you'?re right", r"my mistake", r"i was wrong", r"i apologi[sz]e", r"as an ai", r"language model",
    r"great question", r"i'?m not sure", r"i don'?t know", r"i can'?t be certain", r"mote",
]
_MARKERS = re.compile("|".join(RUBRIC_MARKERS), re.IGNORECASE)


def rubric() -> dict:
    """The rater's rules and a short hash of them; every AI verdict records the hash it was judged under."""
    text = RUBRIC_FILE.read_text(encoding="utf-8") if RUBRIC_FILE.exists() else ""
    return {"text": text, "hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] if text else None}


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from a killed process; the rest is fine
    return out


class PrefStore:
    """One directory of pairs and votes. Everything re-reads the files: they are small and the studio is single-user."""

    def __init__(self, pairs_file: Path = PAIRS_FILE, votes_file: Path = VOTES_FILE):
        self.pairs_file = Path(pairs_file)
        self.votes_file = Path(votes_file)

    # ---- writing ---------------------------------------------------------------------------
    def add_pair(self, messages: Sequence[dict], a: str, b: str, a_source: dict, b_source: dict, origin: str,
                 pair_id: Optional[str] = None) -> dict:
        if a == b:
            raise ValueError("the two replies are identical: nothing to prefer")
        rec = {"id": pair_id or secrets.token_hex(6), "ts": time.time(), "messages": list(messages), "a": a, "b": b,
               "a_source": dict(a_source), "b_source": dict(b_source), "origin": origin}
        _append(self.pairs_file, rec)
        return rec

    def add_vote(self, pair_id: str, rater: str, vote: str, reason: str = "", rubric_hash: Optional[str] = None) -> dict:
        if rater not in RATERS:
            raise ValueError(f"rater must be one of {RATERS}")
        if vote not in VOTES:
            raise ValueError(f"vote must be one of {VOTES}")
        rec = {"pair": pair_id, "ts": time.time(), "rater": rater, "vote": vote, "reason": reason or "", "rubric": rubric_hash}
        _append(self.votes_file, rec)
        return rec

    # ---- reading ---------------------------------------------------------------------------
    def pairs(self) -> List[dict]:
        return _read(self.pairs_file)

    def votes(self) -> List[dict]:
        return _read(self.votes_file)

    def latest_votes(self) -> Dict[str, Dict[str, dict]]:
        """pair id -> rater -> the newest vote of that rater."""
        out: Dict[str, Dict[str, dict]] = {}
        for v in self.votes():
            out.setdefault(v["pair"], {})[v["rater"]] = v
        return out

    # ---- numbers ---------------------------------------------------------------------------
    def summary(self) -> dict:
        pairs = self.pairs()
        latest = self.latest_votes()
        table: Dict[tuple, dict] = {}
        n_votes = {r: 0 for r in RATERS}
        for p in pairs:
            v = latest.get(p["id"], {})
            for r in RATERS:
                if r in v:
                    n_votes[r] += 1
            user = v.get("user")
            if user is None:
                continue
            x, y = _src_key(p["a_source"]), _src_key(p["b_source"])
            key, flipped = (x, y), False
            if y < x:
                key, flipped = (y, x), True
            row = table.setdefault(key, {"a": key[0], "b": key[1], "a_wins": 0, "b_wins": 0, "ties": 0, "both_bad": 0, "n": 0})
            row["n"] += 1
            vote = user["vote"]
            if vote == "tie":
                row["ties"] += 1
            elif vote == "both_bad":
                row["both_bad"] += 1
            elif (vote == "a") != flipped:
                row["a_wins"] += 1
            else:
                row["b_wins"] += 1
        both = [(v["user"]["vote"], v["claude"]["vote"]) for v in latest.values() if "user" in v and "claude" in v]
        decided = [(u, c) for u, c in both if u in ("a", "b") and c in ("a", "b")]
        agree = sum(1 for u, c in decided if u == c)
        return {
            "pairs": len(pairs), "votes": n_votes, "unrated_by_claude": sum(1 for p in pairs if "claude" not in latest.get(p["id"], {})),
            "table": sorted(table.values(), key=lambda r: -r["n"]),
            "agreement": {"n": len(decided), "agree": agree, "rate": agree / len(decided) if decided else None},
            "rubric": rubric()["hash"],
        }

    def disagreements(self) -> List[dict]:
        """Pairs where you and the rater picked different sides (ties and both-bad are listed separately as 'soft')."""
        latest = self.latest_votes()
        out = []
        for p in self.pairs():
            v = latest.get(p["id"], {})
            if "user" not in v or "claude" not in v:
                continue
            u, c = v["user"]["vote"], v["claude"]["vote"]
            if u == c:
                continue
            out.append({"id": p["id"], "messages": p["messages"], "a": p["a"], "b": p["b"], "a_source": p["a_source"],
                        "b_source": p["b_source"], "user": u, "user_reason": v["user"].get("reason", ""),
                        "claude": c, "claude_reason": v["claude"].get("reason", ""),
                        "hard": u in ("a", "b") and c in ("a", "b")})
        out.sort(key=lambda d: (not d["hard"], d["id"]))
        return out

    # ---- the rater's batch -----------------------------------------------------------------
    def export_for_rating(self, out: Path, limit: Optional[int] = None, include_rated: bool = False) -> int:
        """Unrated pairs, most informative first, without sources or the user's vote (the rater is blind)."""
        latest = self.latest_votes()
        rows = []
        for p in self.pairs():
            v = latest.get(p["id"], {})
            if "claude" in v and not include_rated:
                continue
            rows.append((("user" in v), divergence(p["a"], p["b"]), p))
        # your own votes first (they calibrate the rater), then by how different the two replies are
        rows.sort(key=lambda r: (not r[0], -r[1]))
        if limit is not None:
            rows = rows[:limit]
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rub = rubric()
        with open(out, "w", encoding="utf-8") as f:
            for _, d, p in rows:
                f.write(json.dumps({"id": p["id"], "messages": p["messages"], "a": p["a"], "b": p["b"],
                                    "divergence": round(d, 3), "rubric": rub["hash"]}, ensure_ascii=False) + "\n")
        return len(rows)

    def import_verdicts(self, path: Path, rater: str = "claude") -> int:
        """JSONL of {"id", "vote", "reason"}; the rubric hash is stamped from docs/rubric.md as it is now."""
        known = {p["id"] for p in self.pairs()}
        rub = rubric()["hash"]
        n = 0
        for rec in _read(Path(path)):
            if rec.get("id") in known and rec.get("vote") in VOTES:
                self.add_vote(rec["id"], rater, rec["vote"], rec.get("reason", ""), rub)
                n += 1
        return n


def _src_key(src: dict) -> str:
    return f"{src.get('checkpoint', '?')}@{src.get('step', '?')}"


def divergence(a: str, b: str) -> float:
    """0 = identical, ~1 = nothing in common. Edit distance ratio, the length gap, and whether only one
    side trips a rubric marker — the pairs where a verdict carries information."""
    ratio = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    la, lb = max(len(a), 1), max(len(b), 1)
    gap = abs(la - lb) / max(la, lb)
    markers = (1.0 if bool(_MARKERS.search(a)) != bool(_MARKERS.search(b)) else 0.0)
    return min(1.0, 0.6 * (1.0 - ratio) + 0.2 * gap + 0.2 * markers)
