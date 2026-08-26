"""Counterfactual minimal pairs — the same world, one different final action (docs/shape.md § mid).

    python -m mote.sim.counterfactual --out data/sim_cf --n 20000

An ordinary narrative never isolates *why* an answer is what it is: the reader sees one history and one
outcome, and any of the events could have been the load-bearing one. A pair that shares every byte but
the last event, asks the same question, and disagrees on the answer says exactly which event mattered.

It is the construction that fixed the identity pushback set on 2026-08-25 — one template rendered twice
with the values exchanged, so only the claim's truth distinguishes the sides — applied to world state
instead of arithmetic. 2605.17528 (CausalSynth) does the same thing from the other end: generate causal
skeletons from a structural causal model, then pay a large model to realise them in prose. Mote's sim IS
the structural causal model, so the expensive half is free.

No replay-to-step API was needed, which is why this exists at all. Diverting the LAST action leaves the
prefix bit-identical, because the RNG draws that chose it have already happened — only the final sentence
and the answers depending on it change. (A divert at an arbitrary tick still wants that API; so does the
parked PIVOT continuation item.)

Rows are `{"text", "meta"}` for `build_local --text`, written as adjacent pairs so a shuffle keeps them in
the same shard but a window rarely spans both.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .domains import DOMAINS, make_counterfactual, sample_difficulty
from .render import narrative, qa_pairs

LOCALE_WEIGHTS = [("en", 0.6), ("ru", 0.2), ("ja", 0.2)]


def matched_questions(a, b, locale: str) -> List[Tuple[Dict, Dict]]:
    """Questions asked of BOTH worlds whose answers disagree.

    The question builder branches on state — a held object gets "who has it", a loose one gets "where is
    it" — so a changed final action changes which questions exist. Pairing on (qtype, args) keeps only the
    ones that are genuinely the same question, which is the whole point of a minimal pair."""
    qa, qb = qa_pairs(a, locale), qa_pairs(b, locale)
    key = lambda q: (q["qtype"], json.dumps({k: str(v) for k, v in q["args"].items()}, sort_keys=True))
    by_b = {key(q): q for q in qb}
    out = []
    for q in qa:
        m = by_b.get(key(q))
        if m is not None and m["answer"] != q["answer"]:
            out.append((q, m))
    return out


def generate(n: int, seed: int, p_fail: int, domains: Optional[List[str]] = None) -> Tuple[List[Dict], Dict]:
    rng = random.Random(seed)
    doms = domains or [d for d in sorted(DOMAINS) if d != "kinship"]
    rows: List[Dict] = []
    stats = Counter()
    shared: List[float] = []
    s = seed
    while len(rows) < n * 2 and s < seed + n * 60:
        s += 1
        domain = doms[s % len(doms)]
        locale = rng.choices([l for l, _ in LOCALE_WEIGHTS], [w for _, w in LOCALE_WEIGHTS])[0]
        pair = make_counterfactual(domain, s, sample_difficulty(random.Random(s ^ 0x5EED), p_fail))
        stats["seeds"] += 1
        if pair is None:
            stats["no_divert"] += 1
            continue
        a, b = pair
        try:
            da, db = narrative(a, locale), narrative(b, locale)
            matched = matched_questions(a, b, locale)
        finally:
            a.world.close()
            b.world.close()
        if not da or not db or not matched:
            stats["no_matched_question"] += 1
            continue
        q_a, q_b = matched[0]
        pre = sum(1 for x, y in zip(da, db) if x == y)
        shared.append(pre / max(len(da), 1))
        stats["pairs"] += 1
        for doc, q, branch in ((da, q_a, "factual"), (db, q_b, "counterfactual")):
            rows.append({"text": f"{doc}\n\n{q['question']}\n{q['answer']}",
                         "meta": {"domain": domain, "locale": locale, "seed": s, "branch": branch,
                                  "qtype": q["qtype"], "p_fail": p_fail}})
    manifest = {
        "rows": len(rows), "pairs": stats["pairs"], "seeds_tried": stats["seeds"],
        "rejected_no_divert": stats["no_divert"], "rejected_no_matched_question": stats["no_matched_question"],
        "mean_shared_prefix": round(sum(shared) / len(shared), 4) if shared else 0.0,
        "bytes": sum(len(r["text"].encode()) for r in rows), "p_fail": p_fail,
    }
    return rows, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="prefix: writes <out>.jsonl and <out>.json")
    ap.add_argument("--n", type=int, default=20000, help="pairs (two rows each)")
    ap.add_argument("--seed", type=int, default=950_000, help="far from the training and probe ranges")
    ap.add_argument("--p-fail", type=int, default=0, help="match whatever the main generator was built with")
    args = ap.parse_args(argv)
    rows, manifest = generate(args.n, args.seed, args.p_fail)
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
