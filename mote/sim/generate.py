"""Generate the three outputs from simulated traces.

    python -m mote.sim.generate --out data/sim_state --mb 150 --dpo-pairs 20000

Emits (jsonl): <out>.narrative.jsonl {text, meta}, <out>.qa.jsonl {messages, meta},
<out>.dpo.jsonl {prompt, chosen, rejected, meta}, and <out>.gate.json with the diversity stats
half of the 3-part gate (correctness holds by construction; the 50-sample read is manual).
Locale split: en 60% / ru 20% / ja 20%. Domains uniform. Difficulty sampled per trace, in meta.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .domains import DOMAINS, make_trace
from .render import narrative, qa_pairs

LOCALE_WEIGHTS = [("en", 0.6), ("ru", 0.2), ("ja", 0.2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=float, default=150.0, help="approximate narrative+qa megabytes")
    ap.add_argument("--dpo-pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    domains = sorted(DOMAINS)
    f_nar = open(f"{out}.narrative.jsonl", "w", encoding="utf-8")
    f_qa = open(f"{out}.qa.jsonl", "w", encoding="utf-8")
    f_dpo = open(f"{out}.dpo.jsonl", "w", encoding="utf-8")

    written = 0
    n_docs = n_qa = n_dpo = 0
    qtype_counts: Counter = Counter()
    wrongkind_counts: Counter = Counter()
    trigram_counts: Counter = Counter()
    target = int(args.mb * 1e6)
    seed = args.seed
    while written < target or n_dpo < args.dpo_pairs:
        seed += 1
        domain = domains[seed % len(domains)]
        locale = rng.choices([l for l, _ in LOCALE_WEIGHTS], [w for _, w in LOCALE_WEIGHTS])[0]
        trace = make_trace(domain, seed)
        doc = narrative(trace, locale)
        pairs = qa_pairs(trace, locale)
        trace.world.close()  # esper worlds are global; free each one once rendered
        if not doc or not trace.questions:
            continue
        meta = {"domain": domain, "locale": locale, "seed": seed, **trace.difficulty}
        if written < target:
            f_nar.write(json.dumps({"text": doc, "meta": meta}, ensure_ascii=False) + "\n")
            written += len(doc.encode("utf-8"))
            n_docs += 1
        for p in pairs:
            qtype_counts[p["qtype"]] += 1
            words = doc.split()[:50]
            for i in range(len(words) - 2):
                trigram_counts[" ".join(words[i : i + 3])] += 1
            if written < target:
                msg = {"messages": [
                    {"role": "user", "content": f"{doc}\n\n{p['question']}"},
                    {"role": "assistant", "content": p["answer"]},
                ], "meta": {**meta, "qtype": p["qtype"]}}
                f_qa.write(json.dumps(msg, ensure_ascii=False) + "\n")
                written += len(p["question"].encode()) + len(p["answer"].encode()) + len(doc.encode("utf-8"))
                n_qa += 1
            if n_dpo < args.dpo_pairs:
                wrongkind_counts[p["wrong_kind"]] += 1
                f_dpo.write(json.dumps({
                    "prompt": f"{doc}\n\n{p['question']}",
                    "chosen": p["answer"], "rejected": p["wrong"],
                    "meta": {**meta, "qtype": p["qtype"], "wrong_kind": p["wrong_kind"]},
                }, ensure_ascii=False) + "\n")
                n_dpo += 1
    for f in (f_nar, f_qa, f_dpo):
        f.close()

    total_tri = sum(trigram_counts.values())
    top_tri = trigram_counts.most_common(5)
    gate = {
        "docs": n_docs, "qa": n_qa, "dpo": n_dpo, "bytes": written,
        "qtype_counts": dict(qtype_counts), "wrong_kind_counts": dict(wrongkind_counts),
        "distinct_qtypes": len(qtype_counts),
        "top_trigram_share": (top_tri[0][1] / total_tri) if total_tri else 0.0,
        "top_trigrams": top_tri,
    }
    Path(f"{out}.gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False))
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
