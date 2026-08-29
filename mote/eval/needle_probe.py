"""Needle-in-chat probe: a fact stated in turn 1, asked after D bytes of filler turns — does the model
still answer, with folding (docs/context.md) versus plain truncation?

    python -m mote.eval.needle_probe --checkpoint runs/overnight_sft/last.pt [--device cpu] [--distances 512,1024,2048,4096]

For each fact × distance × mode the conversation is: user states the fact, assistant acknowledges,
generic filler exchanges until the distance is reached, then the question. Greedy decoding, ≤ 48
bytes; a hit is the needle word in the reply. With folding, the first user message rides in the
compaction card once the window overflows; with truncation it is gone. Writes needle_probe.json next
to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Dict, List

from ..infer.engine import Engine, GenParams

FACTS = [
    ("My dog's name is Biscuit.", "What is my dog's name?", "biscuit"),
    ("I live in Lisbon.", "Which city do I live in?", "lisbon"),
    ("My favourite colour is green.", "What is my favourite colour?", "green"),
    ("My sister is called Mara.", "What is my sister's name?", "mara"),
    ("I drive a red Fiat.", "What car do I drive?", "fiat"),
    ("My birthday is in March.", "In which month is my birthday?", "march"),
]

FILLER = [
    ("Can you tell me something about rivers?", "Rivers carry fresh water from high ground to the sea, shaping valleys as they go."),
    ("What do bees do all day?", "Bees gather nectar and pollen, tend the hive and keep the brood warm."),
    ("How does bread rise?", "Yeast eats sugars in the dough and gives off gas, which the gluten traps as bubbles."),
    ("Why is the sky blue?", "Air scatters short blue wavelengths of sunlight more than the long red ones."),
    ("What makes a good cup of tea?", "Fresh water just off the boil, enough leaf, and a few minutes of patience."),
    ("Tell me about the moon.", "The Moon circles Earth about once a month and raises the tides as it goes."),
    ("How do trains stay on the rails?", "The wheels are slightly cone-shaped, so they steer themselves back to the centre."),
    ("What is a haiku?", "A three-line poem of five, seven and five syllables, usually about a moment in nature."),
]


def _reply(eng: Engine, messages: List[Dict], mode: str, max_bytes: int = 48) -> str:
    ev: List[dict] = []
    eng.generate(messages, GenParams(temperature=0.0, top_p=1.0, max_bytes=max_bytes), ev.append, threading.Event(),
                 context={"fold": mode})
    return ev[-1]["text"] if ev and ev[-1]["type"] == "done" else ""


def conversation(fact: str, question: str, distance: int) -> List[Dict]:
    msgs = [{"role": "user", "content": fact}, {"role": "assistant", "content": "Noted."}]
    used, i = 0, 0
    while used < distance:
        u, a = FILLER[i % len(FILLER)]
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        used += len(u.encode()) + len(a.encode()) + 4
        i += 1
    msgs.append({"role": "user", "content": question})
    return msgs


def run(eng: Engine, distances: List[int]) -> Dict:
    rows, rates = [], {}
    for d in distances:
        for mode in ("auto", "off"):
            hits = 0
            for fact, q, needle in FACTS:
                a = _reply(eng, conversation(fact, q, d), mode)
                ok = needle in a.lower()
                hits += ok
                rows.append({"distance": d, "mode": mode, "q": q, "a": a, "ok": ok})
            rates[f"{mode}@{d}"] = hits / len(FACTS)
    return {"rates": rates, "n_facts": len(FACTS), "context_limit": eng.cfg.max_seq_len, "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--distances", default="512,1024,2048,4096")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    eng = Engine(args.checkpoint, device=args.device)
    res = run(eng, [int(x) for x in args.distances.split(",")])
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "needle_probe.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    for r in res["rows"]:
        print(f"[{'ok ' if r['ok'] else 'BAD'}] {r['mode']:4s}@{r['distance']:<5d} {r['q']!r:34s} -> {r['a'][:60]!r}")


if __name__ == "__main__":
    main()
