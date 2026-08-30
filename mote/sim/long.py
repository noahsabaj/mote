"""Long simulated narratives — the mid-training mix's long-range dependency source.

    python -m mote.sim.long --out data/sim_long --mb 300 --min-bytes 4000 --max-bytes 16000

OctoLong (2608.05141) makes the case that ordinary long corpora — books, papers, repositories — are long
but *locally coherent*, and therefore scarce in the long-distance dependencies that a long context is
supposed to teach. Their answer was to build code contexts by recursively resolving cross-repository
references with an AST parser and a language server, and swapping ~12 % of a conventional context-extension
mixture for them moved long-range retrieval, state tracking and downstream agentic tasks.

Mote has no language server, and its code corpus (codeparrot-clean) ships file bodies without the repo
structure that would let one be built. But it already has the property from a different direction:
`mote.sim` simulates a world tick by tick and asks questions whose answers depend on *where an entity
ended up*, so an answer is determined by an event that may be thousands of bytes earlier in the narrative
and is contradicted by an earlier one. Raising the tick count turns that into an arbitrarily long
dependency without changing the generator's semantics.

The claim is measured rather than asserted. `dependency_distance` locates the last mention of the gold
answer in the narrative and reports how many bytes of text sit between it and the question; the manifest
carries the distribution, so "long-range" is a number in `<out>.long.json` and not an adjective here.
Also relevant: PRISM (2603.17074 §8.1) found short-context mid-training collapses RULER@128k from 59.09
to 6.46, and Mote's ANNEAL reweighting cuts the long-document share 10.0 % -> 8.6 %. This is what pays
that back, and `needle_auto` is a gate guard so a regression cannot ship regardless.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .domains import DOMAINS, OBJECTS, PEOPLE, ROOMS, make_trace, sample_difficulty
from .render import narrative, qa_pairs

LOCALE_WEIGHTS = [("en", 0.6), ("ru", 0.2), ("ja", 0.2)]


def long_difficulty(rng: random.Random, ticks_lo: int, ticks_hi: int, p_fail: int = 0) -> Dict[str, int]:
    """The ordinary difficulty knobs, opened up as far as the entity pools allow.

    More ticks is what makes the narrative long. More people, rooms and objects is what stops the extra
    ticks being the same two entities moving back and forth, which would be length without dependency —
    but the pools in `domains` are small (8 people, 6 rooms, 8 objects), and asking for more than exists
    raises from `random.sample`. So each knob is capped at its pool and the length comes from ticks.

    One sampler, wider ranges (2026-08-29): this used to be its own dict and so had no `p_fail` — the long
    mix was the flawless-expert corpus the 08-26 reading argued against. Same draw order as before, so a
    long trace at p_fail 0 is the trace it always was."""
    return sample_difficulty(rng, p_fail, people=(4, len(PEOPLE)), rooms=(4, len(ROOMS)),
                             objects=(5, len(OBJECTS)), ticks=(ticks_lo, ticks_hi))


_STOP = {"where", "what", "who", "when", "how", "many", "is", "was", "are", "were", "the", "a", "an",
         "now", "at", "in", "on", "beginning", "end", "does", "did", "have", "has", "it", "of", "and"}


def dependency_distance(doc: str, question: str) -> Optional[int]:
    """Bytes from the *earliest* last-mention of the question's entities to the end of the narrative.

    A gold answer is a whole sentence ("It is in the garden.") and never appears verbatim in the events,
    so the answer string is the wrong anchor — the first version of this measured 0 % coverage for exactly
    that reason. What actually determines the answer is the last event about the entity the question names:
    "Where is the lamp now?" is settled by the final `lamp` event, however long ago that was. When a
    question names several entities the model must reach the furthest of them back, so the binding
    constraint is the *minimum* of their last positions.

    Returns None when no entity name from the question occurs in the narrative — chiefly the ja locale,
    which is unsegmented, so the manifest reports coverage per locale rather than one number."""
    words = [w.strip(".,?!;:'\"") for w in (question or "").lower().split()]
    hits = [doc.lower().rfind(w) for w in words if len(w) >= 3 and w not in _STOP]
    hits = [h for h in hits if h >= 0]
    if not hits:
        return None
    return len(doc[min(hits):].encode("utf-8"))


def generate(mb: float, seed: int, min_bytes: int, max_bytes: int, ticks: Tuple[int, int],
             per_trace: int, p_fail: int = 0) -> Tuple[List[Dict], Dict]:
    rng = random.Random(seed)
    domains = sorted(DOMAINS)
    rows: List[Dict] = []
    dists: List[int] = []
    lens: List[int] = []
    n_cover = n_q = 0
    dom_counts: Counter = Counter()
    loc_q: Counter = Counter()
    loc_cover: Counter = Counter()
    written, target, s = 0, int(mb * 1e6), seed
    tries = 0
    while written < target and tries < 400_000:
        tries += 1
        s += 1
        domain = domains[s % len(domains)]
        locale = rng.choices([l for l, _ in LOCALE_WEIGHTS], [w for _, w in LOCALE_WEIGHTS])[0]
        trace = make_trace(domain, s, long_difficulty(random.Random(s ^ 0x10119), *ticks, p_fail=p_fail))
        try:
            doc = narrative(trace, locale)
            pairs = qa_pairs(trace, locale)
        finally:
            trace.world.close()
        nb = len(doc.encode("utf-8"))
        if not doc or not pairs or not (min_bytes <= nb <= max_bytes):
            continue
        meta = {"domain": domain, "locale": locale, "seed": s, **trace.difficulty}
        # the narrative alone, as a document: long-range structure with no question attached
        rows.append({"text": doc, "meta": {**meta, "kind": "narrative", "bytes": nb}})
        written += nb
        lens.append(nb)
        dom_counts[domain] += 1
        # a few questions appended, so some windows carry the dependency *and* the thing that resolves it
        step = max(len(pairs) // per_trace, 1)
        for p in pairs[::step][:per_trace]:
            n_q += 1
            loc_q[locale] += 1
            d = dependency_distance(doc, p["question"])
            if d is not None:
                n_cover += 1
                loc_cover[locale] += 1
                dists.append(d)
            body = f"{doc}\n\n{p['question']}\n{p['answer']}"
            rows.append({"text": body, "meta": {**meta, "kind": "qa", "qtype": p["qtype"],
                                                "bytes": len(body.encode('utf-8')),
                                                **({"dep_bytes": d} if d is not None else {})}})
            written += len(body.encode("utf-8"))
    dists.sort()

    def pct(q: float) -> int:
        return dists[min(int(q * len(dists)), len(dists) - 1)] if dists else 0

    manifest = {
        "rows": len(rows), "bytes": written, "narratives": sum(1 for r in rows if r["meta"]["kind"] == "narrative"),
        "qa": sum(1 for r in rows if r["meta"]["kind"] == "qa"),
        "doc_bytes": {"mean": int(statistics.mean(lens)) if lens else 0,
                      "p50": int(statistics.median(lens)) if lens else 0, "max": max(lens) if lens else 0},
        "dependency_distance_bytes": {"n": len(dists), "coverage": n_cover / max(n_q, 1),
                                      "p50": pct(0.5), "p90": pct(0.9), "max": dists[-1] if dists else 0,
                                      "over_1k": sum(d > 1024 for d in dists) / max(len(dists), 1),
                                      "over_4k": sum(d > 4096 for d in dists) / max(len(dists), 1)},
        "coverage_per_locale": {l: loc_cover[l] / loc_q[l] for l in sorted(loc_q)},
        "per_domain": dict(dom_counts), "ticks": list(ticks),
    }
    return rows, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="prefix: writes <out>.jsonl and <out>.json")
    ap.add_argument("--mb", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=900_000, help="far from the training and probe seed ranges")
    ap.add_argument("--min-bytes", type=int, default=4000, help="drop narratives shorter than this — the point is length")
    ap.add_argument("--max-bytes", type=int, default=16000, help="one window is 16384 bytes; a document should fit in one")
    ap.add_argument("--ticks", default="60,220", help="events per trace, lo,hi (the ordinary generator uses 4,18)")
    ap.add_argument("--per-trace", type=int, default=2, help="questions appended per narrative")
    ap.add_argument("--p-fail", type=int, default=0, help="percentage of scripted actions that are deliberately illegal, exactly as mote.sim.generate's flag — regenerate the long mix with the SAME value as the short traces (2026-08-29); 0 keeps every long trace as it was")
    args = ap.parse_args(argv)
    lo, hi = (int(x) for x in args.ticks.split(","))
    rows, manifest = generate(args.mb, args.seed, args.min_bytes, args.max_bytes, (lo, hi), args.per_trace, args.p_fail)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(f"{out}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
