"""Prefix-cache probe: a scripted multi-turn conversation through the serving engine, warm vs cold.

    python -m mote.eval.prefix_probe --checkpoint runs/overnight_sft2/last.pt --turns 40 --device cpu

Per turn: prompt bytes, bytes reused from the cache, warm read time, cold read time, the number of
chunk cuts that moved and the largest next-byte logit difference between the warm continuation and a
cold read of the same prompt (Engine._verify_prefix), and whether the greedy reply differs from the one
a cache-less engine produces. Decided 2026-08-23 (docs/context.md): the cache ships with this number
measured, not assumed.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from ..serve.engine import Engine, GenParams

QUESTIONS = [
    "Hi! What are you?", "What can you help me with?", "Name three colours.", "What is two plus two?",
    "Write one sentence about the sea.", "Which is bigger, a cat or a mouse?", "Say something about bread.",
    "What day comes after Monday?", "Describe a tree in one line.", "What do birds do?", "Tell me about rain.",
    "How many legs does a dog have?", "What is a river?", "Give me a word that rhymes with cat.",
    "What colour is the sky?", "Name a fruit.", "What is a book for?", "Say hello in one word.",
    "What do you know about music?", "Is fire hot or cold?", "What is a city?", "Tell me about the moon.",
    "What is your favourite letter?", "Describe snow.", "What do fish do?", "Name a vegetable.",
    "What is a mountain?", "What sound does a cow make?", "Tell me about the wind.", "What is milk?",
    "What is a chair for?", "Name a season.", "What is the opposite of up?", "Tell me about a garden.",
    "What is a clock?", "Name a number bigger than ten.", "What do bees make?", "Describe a house.",
    "What is a friend?", "Say goodbye.",
]


def run_turn(eng: Engine, messages, params, context):
    evs = []
    eng.generate(messages, params, evs.append, threading.Event(), context=context)
    start = next(e for e in evs if e["type"] == "start")
    done = next(e for e in evs if e["type"] == "done")
    check = next((e["prefix_check"] for e in evs if e["type"] == "diagnostics" and "prefix_check" in e), None)
    return start, done, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/overnight_sft2/last.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--max-bytes", type=int, default=48)
    ap.add_argument("--out", default=None, help="JSON path (default: next to the checkpoint)")
    args = ap.parse_args()

    warm = Engine(args.checkpoint, device=args.device)
    cold = Engine(args.checkpoint, device=args.device, prefix_cache_mb=0)
    if args.device != "cpu":  # the Triton JIT (tens of seconds on first use) is startup cost, not a turn
        print(f"warm-up: {warm.warmup():.1f} s + {cold.warmup():.1f} s", flush=True)
    params = GenParams(temperature=0.0, max_bytes=args.max_bytes, n_candidates=0)
    messages, prev, rows = [], None, []
    for t, q in enumerate(QUESTIONS[: args.turns]):
        messages.append({"role": "user", "content": q})
        ctx = {"fold": "auto", "verify_prefix": True, "prev": prev}
        t0 = time.perf_counter()
        s, d, chk = run_turn(warm, messages, params, ctx)
        warm_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, d_cold, _ = run_turn(cold, messages, params, {"fold": "auto", "prev": prev})
        cold_s = time.perf_counter() - t0
        prev = {"from": s["fold"]["from"], "card": s["fold"]["card"]} if s.get("fold") else None
        row = {
            "turn": t + 1, "prompt_bytes": s["prompt_bytes"], "reused": s["prefix"]["reused"],
            "prefilled": s["prefix"]["prefilled"], "warm_prefill_ms": s["prefix"]["prefill_ms"],
            "cold_prefill_ms": chk["cold_ms"] if chk else None,
            "boundary_flips": chk["boundary_flips"] if chk else 0,
            "max_logit_diff": chk["max_logit_diff"] if chk else 0.0,
            "same_reply": d["text"] == d_cold["text"], "folded": bool(s.get("fold")),
            "warm_turn_s": warm_s, "cold_turn_s": cold_s,
        }
        rows.append(row)
        print(f"turn {row['turn']:2d}  prompt {row['prompt_bytes']:5d} B  reused {row['reused']:5d}  read {row['prefilled']:4d}  "
              f"warm {row['warm_prefill_ms']:7.0f} ms  cold {row['cold_prefill_ms'] or 0:7.0f} ms  flips {row['boundary_flips']}  "
              f"dlogit {row['max_logit_diff']:.2e}  same reply {row['same_reply']}  fold {row['folded']}", flush=True)
        messages.append({"role": "assistant", "content": d["text"]})

    n = len(rows)
    summary = {
        "turns": n, "reused_fraction": sum(r["reused"] for r in rows) / max(sum(r["prompt_bytes"] for r in rows), 1),
        "turns_with_flips": sum(1 for r in rows if r["boundary_flips"]),
        "max_logit_diff": max(r["max_logit_diff"] for r in rows),
        "replies_differ": sum(1 for r in rows if not r["same_reply"]),
        "warm_prefill_ms_mean": sum(r["warm_prefill_ms"] for r in rows) / n,
        "cold_prefill_ms_mean": sum(r["cold_prefill_ms"] or 0 for r in rows) / n,
        # the worst turn is the availability number (a harness times out on the tail, not the mean)
        "warm_prefill_ms_max": max(r["warm_prefill_ms"] for r in rows),
        "cold_prefill_ms_max": max(r["cold_prefill_ms"] or 0 for r in rows),
        "warm_turn_s_max": max(r["warm_turn_s"] for r in rows),
        "worst_turn": max(rows, key=lambda r: r["warm_prefill_ms"])["turn"],
        "rows_copied_in": warm.prefix_cache.rows_copied_in,  # arena rows hydrated from CPU pages (0 = every turn was hot)
        "device": args.device, "checkpoint": args.checkpoint,
    }
    print(json.dumps(summary, indent=1))
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "prefix_probe.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
