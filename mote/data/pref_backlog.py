"""Fill the preference store with pairs worth rating, so the gate is reachable at all.

    python -m mote.data.pref_backlog --checkpoint runs/flagship_sft/last.pt --n 200 --temperature 0.9
    mote prefs export --out data/prefs/to_rate.jsonl      # hands the unrated ones to Claude, in session
    mote prefs import data/prefs/verdicts.jsonl

docs/prefs.md gates the first prefs DPO round at ~1000 rated pairs with ~150 of Noah's. Two days after the
store was built it held 2 votes, both `both_bad` — the gate is unreachable at the rate a pair arrives from
ordinary use, because a pair costs two generations and a deliberate comparison. Three things fix that and
this is the third: marks (mote.train.kto) make single replies usable, Retry/Compare/Arena already mine
pairs as a side-effect (chat.svelte.ts), and this generates a backlog for the AI rater so Noah's ~150 can
be the calibration set rather than the bulk.

Pairs are two samples of the *same* prompt from the same checkpoint at a temperature high enough to
separate them — that is the pair source docs/prefs.md already names for when no challenger is loaded. The
prompts lean on what docs/rubric.md actually judges: hedging, sycophancy, made-up specifics, refusing to
say "I don't know". Pass --prompts to use your own bank instead of the starter one.

Nothing here votes. The pairs land unrated; `mote prefs export` is what hands them out.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Dict, List

from ..serve.engine import Engine, GenParams
from ..serve.identity import with_system_card
from ..serve.prefs import PrefStore

# The rubric's own concerns, one prompt each: a reply can be good or bad here in ways worth a rater's time.
STARTER_PROMPTS: List[str] = [
    "What is the capital of Australia?",
    "Who is Marie Curie?",
    "How many moons does Mars have?",
    "Explain what a compiler does, briefly.",
    "What happened in 1927?",
    "Is it true that lightning never strikes the same place twice?",
    "What's the population of Reykjavik?",
    "Summarise how rain forms in two sentences.",
    "What does the word 'quixotic' mean?",
    "Give me one fact about octopuses.",
    "What is 17 times 24?",
    "Why is the sky blue?",
    "Who wrote 'One Hundred Years of Solitude'?",
    "What is the difference between weather and climate?",
    "Name a country that borders Peru.",
    "How does a fridge stay cold?",
    "What year did the Berlin Wall come down?",
    "Is a tomato a fruit or a vegetable?",
    "What is the tallest mountain in Africa?",
    "Explain recursion to someone who has never programmed.",
]


def sample_pairs(eng: Engine, prompts: List[str], temperature: float, max_bytes: int) -> List[Dict]:
    """Two independent samples of each prompt; a prompt whose two samples match is dropped (nothing to prefer)."""
    n_params = eng.info()["params"]
    out = []
    for q in prompts:
        msgs = [{"role": "user", "content": q}]
        replies = []
        for _ in range(2):
            ev: List[dict] = []
            eng.generate(with_system_card(msgs, n_params),
                         GenParams(temperature=temperature, top_p=0.9, max_bytes=max_bytes, n_candidates=0),
                         ev.append, threading.Event())
            replies.append(ev[-1]["text"] if ev and ev[-1]["type"] == "done" else "")
        a, b = replies
        if not a.strip() or not b.strip() or a == b:
            continue
        out.append({"messages": msgs, "a": a, "b": b})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", default=None, help="JSONL of {prompt} or a plain text file, one per line; default: the starter bank")
    ap.add_argument("--n", type=int, default=100, help="how many pairs to aim for (prompts are cycled)")
    ap.add_argument("--temperature", type=float, default=0.9, help="high enough that the two samples differ")
    ap.add_argument("--max-bytes", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the pairs instead of storing them")
    args = ap.parse_args(argv)

    if args.prompts:
        raw = [l for l in Path(args.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
        prompts = [json.loads(l)["prompt"] if l.lstrip().startswith("{") else l.strip() for l in raw]
    else:
        prompts = STARTER_PROMPTS
    cycled = [prompts[i % len(prompts)] for i in range(args.n)]

    eng = Engine(args.checkpoint, device=args.device)
    pairs = sample_pairs(eng, cycled, args.temperature, args.max_bytes)
    src = {"checkpoint": Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).name,
           "step": int(eng.info().get("checkpoint", {}).get("step", 0) or 0), "engine": "current",
           "params": {"temperature": args.temperature, "top_p": 0.9, "max_bytes": args.max_bytes}}
    if args.dry_run:
        for p in pairs[:5]:
            print(f"Q: {p['messages'][-1]['content']}\n  A: {p['a'][:110]}\n  B: {p['b'][:110]}")
        print(json.dumps({"would_store": len(pairs), "of": args.n}, indent=1))
        return
    store = PrefStore()
    for p in pairs:
        store.add_pair(p["messages"], p["a"], p["b"], src, src, origin="backlog")
    print(json.dumps({"stored": len(pairs), "of": args.n, "dropped_identical": args.n - len(pairs),
                      **store.summary() | {"table": "..."}}, indent=1, default=str))


if __name__ == "__main__":
    main()
