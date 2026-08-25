"""On-Policy Replay: keep a stage's own good answers so the next stage cannot quietly forget them.

    python -m mote.data.replay --checkpoint runs/flagship_sft/last.pt --prompts data/replay/identity.jsonl \
        --out data/replay/sft1_kept.jsonl --budget 300 --temperature 0.0

The pipeline's guards (docs/shape.md: "identity/hold/concede and chat val no regression") *detect* a stage
undoing the one before it. They do not prevent it, and there is nothing in the pipeline that does.

2605.29495 gives the cheap mechanism, and its finding is that the active ingredient is the on-policy
distribution rather than the response quality: roll the previous checkpoint out on a small budget of its
own prompts, throw away the generations that fail their check, and replay the survivors as ordinary SFT
examples mixed into the next stage. No teacher, no auxiliary loss, no distillation — the replayed text is
the model's own, so it costs nothing to stay near. They report |BWT| down 46% at a 10% replay budget and
most of that at 1%.

A prompt record is {"messages": [...], "expect": [...]?, "forbid": [...]?}:

    expect   pass needs at least one of these substrings present (the sim's answer, the model's name)
    forbid   pass needs none of them present — this is how the negative class survives a later stage:
             a neutral question whose answer must NOT contain the identity card or a pushback template

Both are optional; a record with neither keeps every non-empty generation, which is the "replay whatever
it says" ablation the paper compares against (Vanilla Replay) and which it finds is already strong.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

from ..serve.engine import Engine, GenParams
from ..serve.identity import with_system_card


def _hit(text: str, needle: str) -> bool:
    return re.search(r"(?<![\w.])" + re.escape(needle.lower()) + r"(?![\w])", text.lower()) is not None


def passes(reply: str, rec: Dict) -> bool:
    if not reply.strip():
        return False
    expect = rec.get("expect") or []
    forbid = rec.get("forbid") or []
    if expect and not any(_hit(reply, e) for e in expect):
        return False
    if any(_hit(reply, f) for f in forbid):
        return False
    return True


def replay(eng: Engine, records: List[Dict], budget: int, temperature: float, max_bytes: int,
           n_candidates: int = 0) -> Dict:
    """Roll out, filter, and return the survivors as {messages, reply} plus a report."""
    kept, tried = [], 0
    n_params = eng.info()["params"]
    for rec in records[:budget]:
        tried += 1
        ev: List[dict] = []
        eng.generate(with_system_card(rec["messages"], n_params),
                     GenParams(temperature=temperature, top_p=1.0 if temperature <= 0 else 0.9,
                               max_bytes=max_bytes, n_candidates=n_candidates),
                     ev.append, threading.Event())
        reply = ev[-1]["text"] if ev and ev[-1]["type"] == "done" else ""
        if passes(reply, rec):
            kept.append({"messages": rec["messages"], "reply": reply, "src": "replay"})
    return {"rows": kept, "tried": tried, "kept": len(kept),
            "pass_rate": len(kept) / tried if tried else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="the stage whose behaviour is being preserved")
    ap.add_argument("--prompts", required=True, help='JSONL of {messages, expect?, forbid?}')
    ap.add_argument("--out", required=True, help="JSONL of {messages, reply} to mix into the next stage")
    ap.add_argument("--budget", type=int, default=300, help="how many prompts to roll out (the paper's 1-10% is plenty)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; the replayed text is the model's own either way")
    ap.add_argument("--max-bytes", type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    records = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    eng = Engine(args.checkpoint, device=args.device)
    res = replay(eng, records, args.budget, args.temperature, args.max_bytes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in res["rows"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"} | {"out": str(out)}, indent=1))


if __name__ == "__main__":
    main()
