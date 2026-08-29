"""Sim-QA probe: exact-match on held-out simulated worlds (docs/shape.md § pipeline, the mid-training gate
and the RL headroom numbers).

    python -m mote.eval.sim_probe --checkpoint runs/x/last.pt [--n 120] [--k 8] [--device cpu] [--out ...]

Worlds come from `mote.sim` with seeds from --seed-base (5 000 000) upward — the generator's training data
uses seeds 1..N from `--seed 0`, so nothing here was seen in training. Each item is a narrative plus one
question rendered in en / ru / ja; the gold answer is the sim's own. Scores:

  em          greedy reply equals the gold answer after normalisation (case, spaces, final punctuation)
  contains    the gold answer occurs in the greedy reply (lenient)
  pass@1/@k   with --k > 1: k sampled replies at --temperature; pass@k = any of them matches. pass@1 is
              the greedy EM. The RLVR start gate reads these two numbers (pass@1 < 0.5, pass@k − pass@1 ≥ 0.2).

Per-domain and per-locale breakdowns are reported; rows carry every question, gold and reply.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from ..serve.engine import Engine, GenParams
from ..serve.identity import with_system_card
from ..sim.domains import DOMAINS, make_trace, sample_difficulty
from ..sim.render import narrative, qa_pairs

SEED_BASE = 5_000_000
_PUNCT = "。．.!！?？、,;:"


def normalize(s: str) -> str:
    s = s.strip().casefold()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(_PUNCT).strip()


def match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def contains(pred: str, gold: str) -> bool:
    g = normalize(gold)
    return bool(g) and g in normalize(pred)


def heldout_items(n: int, locales: List[str], seed_base: int = SEED_BASE, per_trace: int = 2,
                  p_fail: int = 0) -> List[Dict]:
    """n questions over fresh worlds: domains round-robin by seed, locales round-robin by item.

    `p_fail` must MATCH the rate the training data was built at (signed 2026-08-26). The probe used to
    build its worlds at the default 0 whatever the training data did, so it could never contain a failure
    the model has to notice — it measured a world the model no longer trains on. Held out remains held
    out either way: the seeds start at SEED_BASE, far above the generator's."""
    domains = sorted(DOMAINS)
    items: List[Dict] = []
    seed = seed_base
    while len(items) < n:
        seed += 1
        domain = domains[seed % len(domains)]
        locale = locales[len(items) % len(locales)]
        trace = make_trace(domain, seed, sample_difficulty(random.Random(seed ^ 0x5EED), p_fail))
        try:
            doc = narrative(trace, locale)
            pairs = qa_pairs(trace, locale)
        finally:
            trace.world.close()
        if not doc or not pairs:
            continue
        # spread question types: take every len(pairs)//per_trace-th pair
        step = max(len(pairs) // per_trace, 1)
        for p in pairs[::step][:per_trace]:
            if len(items) >= n:
                break
            items.append({"domain": domain, "locale": locale, "seed": seed, "qtype": p["qtype"],
                          "prompt": f"{doc}\n\n{p['question']}", "gold": p["answer"]})
    return items


def _reply(eng: Engine, prompt: str, temperature: float, max_bytes: int) -> str:
    ev: List[dict] = []
    msgs = with_system_card([{"role": "user", "content": prompt}], eng.info()["params"])
    eng.generate(msgs, GenParams(temperature=temperature, top_p=1.0, max_bytes=max_bytes), ev.append, threading.Event())
    return ev[-1]["text"] if ev and ev[-1]["type"] == "done" else ""


def run(eng: Engine, items: List[Dict], k: int = 1, temperature: float = 1.0, max_bytes: int = 48) -> Dict:
    rows = []
    by_dom: Dict[str, List[int]] = defaultdict(list)
    by_loc: Dict[str, List[int]] = defaultdict(list)
    n_em = n_ct = n_passk = 0
    for it in items:
        greedy = _reply(eng, it["prompt"], 0.0, max_bytes)
        em, ct = match(greedy, it["gold"]), contains(greedy, it["gold"])
        samples = [_reply(eng, it["prompt"], temperature, max_bytes) for _ in range(k)] if k > 1 else []
        passk = em or any(match(s, it["gold"]) for s in samples)
        n_em += em
        n_ct += ct
        n_passk += passk
        by_dom[it["domain"]].append(int(em))
        by_loc[it["locale"]].append(int(em))
        rows.append({**{k_: it[k_] for k_ in ("domain", "locale", "seed", "qtype", "gold")}, "reply": greedy, "em": em, "contains": ct,
                     **({"samples": samples, "pass_k": passk} if k > 1 else {})})
    n = max(len(items), 1)
    out = {"n": len(items), "em": n_em / n, "contains": n_ct / n, "pass_at_1": n_em / n,
           "per_domain": {d: sum(v) / len(v) for d, v in sorted(by_dom.items())},
           "per_locale": {l: sum(v) / len(v) for l, v in sorted(by_loc.items())}}
    if k > 1:
        out.update({"k": k, "temperature": temperature, f"pass_at_{k}": n_passk / n, "headroom": n_passk / n - n_em / n})
    out["rows"] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--k", type=int, default=1, help="samples per question for pass@k (1 = greedy EM only)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--locales", default="en,ru,ja")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--p-fail", type=int, default=0, help="build the held-out worlds at this failure rate; MATCH the training data or the probe measures a world the model does not train on (docs/research/midtraining-2026-08-26.md)")
    ap.add_argument("--max-bytes", type=int, default=48)
    ap.add_argument("--out", default=None, help="default: sim_probe.json next to the checkpoint")
    args = ap.parse_args(argv)
    items = heldout_items(args.n, [l.strip() for l in args.locales.split(",") if l.strip()], args.seed_base,
                          p_fail=args.p_fail)
    eng = Engine(args.checkpoint, device=args.device)
    res = run(eng, items, k=args.k, temperature=args.temperature, max_bytes=args.max_bytes)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "sim_probe.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, ensure_ascii=False))
    for r in res["rows"][:20]:
        print(f"[{'ok ' if r['em'] else 'BAD'}] {r['domain']:9s} {r['locale']} {r['qtype']:16s} gold={r['gold']!r:20s} reply={r['reply'][:60]!r}")


if __name__ == "__main__":
    main()
