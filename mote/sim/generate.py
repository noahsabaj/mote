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

from .domains import DOMAINS, make_trace, sample_difficulty
from .render import lexical_swap, narrative, qa_pairs

LOCALE_WEIGHTS = [("en", 0.6), ("ru", 0.2), ("ja", 0.2)]
LOCALES = [l for l, _ in LOCALE_WEIGHTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=float, default=150.0, help="approximate narrative+qa megabytes")
    ap.add_argument("--dpo-pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-fail", type=int, default=0, help="percentage of scripted actions that are deliberately illegal, so their refusal carries state (docs/research/midtraining-2026-08-26.md). 0 reproduces every pre-2026-08-26 trace exactly; sweep 5/15/30 against the sim probe before choosing")
    ap.add_argument("--parallel-frac", type=float, default=0.0, help="share of worlds rendered in ALL THREE locales rather than one. Identical world state in three surface forms is Allen-Zhu 2309.14316's diversity-of-form argument; 2603.29026 measured parallel data as having minimal effect on cross-lingual ALIGNMENT, which is a different claim, so this is deliberately a fraction and not the default")
    ap.add_argument("--swap-frac", type=float, default=0.0, help="share of English documents with a fraction of their entity words swapped for the ru/ja equivalents (LINK, 2605.23885: up to 2x speedup to equivalent performance from a bilingual vocabulary alone — and the renderer already is one). Cheaper than parallel rendering and keeps world diversity intact")
    ap.add_argument("--swap-rate", type=float, default=0.15, help="fraction of swappable words replaced within a swapped document")
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
        parallel = rng.random() < args.parallel_frac
        locales = LOCALES if parallel else [rng.choices(LOCALES, [w for _, w in LOCALE_WEIGHTS])[0]]
        trace = make_trace(domain, seed, sample_difficulty(random.Random(seed ^ 0x5EED), args.p_fail))
        rendered = [(l, narrative(trace, l), qa_pairs(trace, l)) for l in locales]
        trace.world.close()  # esper worlds are global; free each one once rendered
        if not trace.questions or not all(d for _l, d, _p in rendered):
            continue
        for locale, doc, pairs in rendered:
            # LINK-style substitution, English only: the point is to put another script inside an
            # otherwise-English context, which is not a thing you can do to a document already in it.
            swapped = locale == "en" and rng.random() < args.swap_frac
            if swapped:
                doc = lexical_swap(doc, rng, args.swap_rate)
            meta = {"domain": domain, "locale": locale, "seed": seed,
                    "parallel": parallel, "swapped": swapped, **trace.difficulty}
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
        "p_fail": args.p_fail, "parallel_frac": args.parallel_frac,
        "swap_frac": args.swap_frac, "swap_rate": args.swap_rate,
        "top_trigram_share": (top_tri[0][1] / total_tri) if total_tri else 0.0,
        "top_trigrams": top_tri,
    }
    Path(f"{out}.gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False))
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
