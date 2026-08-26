"""Reading probe: can the model copy an answer span out of a passage placed in the user turn?

    python -m mote.eval.read_probe --checkpoint runs/overnight_sft/last.pt [--n 100] [--device cpu]

The first gate for search (docs/search.md): a model that cannot read a ~1 KB passage and copy the span
cannot use search results either. SQuAD v1.1 validation (rajpurkar/squad via `datasets`, cached),
passages of at most 1024 bytes, greedy decoding, replies of at most 48 bytes. Scores exact match and
token F1 with the SQuAD normalisation against all gold answers, and the same questions *without* the
passage as the baseline, so the gain from reading is visible. Writes read_probe.json next to the
checkpoint.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import string
import threading
from pathlib import Path
from typing import Dict, List

from ..serve.engine import Engine, GenParams
from ..serve.identity import with_system_card

MAX_PASSAGE_BYTES = 1024
MAX_REPLY_BYTES = 48


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def best(pred: str, golds: List[str]):
    em = max(float(normalize(pred) == normalize(g)) for g in golds)
    return em, max(f1(pred, g) for g in golds)


def load_items(n: int, seed: int) -> List[Dict]:
    from datasets import load_dataset  # lazy: only the probe needs it

    ds = load_dataset("rajpurkar/squad", split="validation")
    rows = [r for r in ds if len(r["context"].encode("utf-8")) <= MAX_PASSAGE_BYTES]
    random.Random(seed).shuffle(rows)
    return [{"id": r["id"], "passage": r["context"], "question": r["question"], "answers": list(r["answers"]["text"])} for r in rows[:n]]


def _reply(eng: Engine, content: str) -> str:
    ev: List[dict] = []
    msgs = with_system_card([{"role": "user", "content": content}], eng.info()["params"])
    eng.generate(msgs, GenParams(temperature=0.0, top_p=1.0, max_bytes=MAX_REPLY_BYTES, n_candidates=0), ev.append, threading.Event())
    text = ev[-1]["text"] if ev and ev[-1]["type"] == "done" else ""
    return text.strip().split("\n")[0]


def prompt_for(item: Dict) -> str:
    """The passage-grounded prompt. Named so `mote.eval.proxy` scores the same prompt this probe asks."""
    return f"Read this and answer in a few words.\n\n{item['passage']}\n\nQuestion: {item['question']}"


def run(eng: Engine, items: List[Dict]) -> Dict:
    rows = []
    tot = {"em": 0.0, "f1": 0.0, "em_base": 0.0, "f1_base": 0.0}
    for it in items:
        with_p = _reply(eng, prompt_for(it))
        without = _reply(eng, f"Answer in a few words. {it['question']}")
        em, f = best(with_p, it["answers"])
        emb, fb = best(without, it["answers"])
        tot["em"] += em
        tot["f1"] += f
        tot["em_base"] += emb
        tot["f1_base"] += fb
        rows.append({"id": it["id"], "q": it["question"], "gold": it["answers"][:3], "with_passage": with_p, "without": without, "em": em, "f1": round(f, 3)})
    k = max(1, len(items))
    return {"n": len(items), "exact_match": tot["em"] / k, "f1": tot["f1"] / k,
            "exact_match_no_passage": tot["em_base"] / k, "f1_no_passage": tot["f1_base"] / k, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="default: read_probe.json next to the checkpoint")
    args = ap.parse_args(argv)
    items = load_items(args.n, args.seed)
    eng = Engine(args.checkpoint, device=args.device)
    res = run(eng, items)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "read_probe.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    for r in res["rows"][:25]:
        print(f"[{'ok ' if r['em'] else 'BAD'}] {r['q'][:60]!r:62s} gold={r['gold'][0]!r:28s} -> {r['with_passage'][:40]!r}")


if __name__ == "__main__":
    main()
