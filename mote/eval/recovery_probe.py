"""Recovery probe — does the model do something *else* after a refusal? (docs/shape.md § mid)

    python -m mote.eval.recovery_probe --checkpoint runs/x/last.pt [--n 60]

Why this exists. `mote/sim/tasks.py` refuses an illegal action and answers with one of three strings —
"Nothing happened.", "Unknown action.", "No moves left." — and until 2026-08-26 not one of those appeared
in any of the 20,000 expert traces. The corpus showed a flawless expert, so RLVR-1 would have met its
first refusal having never seen one, with no learned response to it. 2608.20314 (MidTool) and 2607.12463
both single out *recovery from incomplete information* as the thing mid-training should teach, and this
measures whether it was.

The measurement is behavioural and deliberately narrow. Give the model a trace that has just hit a
refusal and score its next call:

  repeat        it issued the same call again — the failure mode, and the one a model with no exposure
                to refusals falls into, because nothing in its context says the call was rejected
  other         a DIFFERENT action that the environment's own parser accepts
  unparseable   text that is not an action at all
  none          it produced nothing

`recovery_rate` = other / (other + repeat + unparseable). Parseability is in the denominator on purpose.
The first version of this scored `other` for anything that was not a verbatim repeat, and a 31.6M base
model that has never seen a tool trace scored a perfect 1.000 on it — its noise never coincidentally
equalled the refused call. A metric a garbage model maxes out is decorative. `mote.sim.tasks.parse_action`
is the environment's real parser, so "did it emit an action at all" is answered by the thing that will
answer it during RLVR-1.

What this still does NOT do is check whether the parseable action is *applicable* from the refused state.
That needs to replay the world to the refusal point — the same replay-to-step API the counterfactual
generator and the parked PIVOT item both want and none of them has. Until then `other` is an upper bound.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List

from ..infer.engine import Engine, GenParams
from ..identity import with_system_card
from ..sim.domains import DOMAINS, make_trace, sample_difficulty
from ..sim.tasks import TEXT as TASK_STRINGS

SEED_BASE = 7_000_000  # far above the generator's and the sim probe's ranges
MAX_REPLY = 64


def build_items(n: int, locales: List[str], seed_base: int = SEED_BASE) -> List[Dict]:
    """A prompt that ends in a refusal, plus the set of actions that would actually work from there.

    Built from the sim rather than from recorded traces so the item is fresh at every run and the legal
    set is computed from the true state instead of guessed from text."""
    items: List[Dict] = []
    doms = [d for d in sorted(DOMAINS) if d != "kinship"]  # kinship has no agent actions
    seed = seed_base
    while len(items) < n:
        seed += 1
        domain = doms[seed % len(doms)]
        locale = locales[len(items) % len(locales)]
        rng = random.Random(seed)
        trace = make_trace(domain, seed, sample_difficulty(random.Random(seed ^ 0x5EED), p_fail=40))
        try:
            from ..sim.render import narrative

            doc = narrative(trace, locale)
            failed = [e for e in trace.events if e.kind == "failed"]
            names = sorted(trace.world.names.values())
        finally:
            trace.world.close()
        if not doc or not failed:
            continue
        e = failed[len(failed) // 2]
        who = e.data.get("who") or e.data.get("buyer")
        refusal = TASK_STRINGS[locale]["nothing"]
        # the narrative up to and including the refused attempt, then the tool's own refusal string
        items.append({
            "domain": domain, "locale": locale, "seed": seed,
            "prompt": f"{doc}\n\n{TASK_STRINGS[locale]['instr']}",
            "refused_call": f"{who}: {e.data['kind']} {e.data.get('obj') or e.data.get('goods') or ''}".strip(),
            "refusal": refusal, "names": names,
        })
    return items


def _parser_for(item: Dict):
    """The environment's own parser, minus the world lookup: a reply counts as an action when it names a
    known actor and a known verb in the domain's grammar. Rebuilding the world here would cost a trace per
    item for no extra signal — the question is "is this an action", not "is it applicable"."""
    verbs = {"household": ("move to", "take", "put", "drop"), "inventory": ("buy", "harvest", "gather"),
             "schedule": ("book", "move")}[item["domain"]]
    names = {n.lower() for n in item["names"]}

    def parses(norm: str) -> bool:
        who, _sep, rest = norm.partition(":")
        return who.strip() in names and any(v in rest for v in verbs)

    return parses


def classify(reply: str, refused: str, parses) -> str:
    """`repeat` / `other` / `unparseable` / `none`, on what the model writes after the refusal."""
    body = reply.strip().split("\n")[0].strip()
    if not body:
        return "none"
    norm = " ".join(body.lower().replace("sim:", "").split())
    ref = " ".join(refused.lower().split())
    if norm == ref or norm.startswith(ref):
        return "repeat"
    return "other" if parses(norm) else "unparseable"


def run(eng: Engine, items: List[Dict]) -> Dict:
    rows, counts = [], Counter()
    for it in items:
        msgs = with_system_card([
            {"role": "user", "content": it["prompt"]},
            {"role": "assistant", "content": f"sim: {it['refused_call']}"},
            {"role": "user", "content": it["refusal"]},
        ], eng.info()["params"])
        ev: List[dict] = []
        eng.generate(msgs, GenParams(temperature=0.0, top_p=1.0, max_bytes=MAX_REPLY),
                     ev.append, threading.Event())
        reply = (ev[-1]["text"] if ev and ev[-1]["type"] == "done" else "").strip()
        kind = classify(reply, it["refused_call"], _parser_for(it))
        counts[kind] += 1
        rows.append({**{k: it[k] for k in ("domain", "locale", "seed")},
                     "refused": it["refused_call"], "reply": reply, "kind": kind})
    acted = counts["repeat"] + counts["other"] + counts["unparseable"]
    return {"n": len(items), "recovery_rate": counts["other"] / acted if acted else 0.0,
            "repeat_rate": counts["repeat"] / acted if acted else 0.0,
            "unparseable_rate": counts["unparseable"] / acted if acted else 0.0,
            "no_action_rate": counts["none"] / max(len(items), 1),
            "counts": dict(counts), "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--locales", default="en,ru,ja")
    ap.add_argument("--out", default=None, help="default: recovery_probe.json next to the checkpoint")
    args = ap.parse_args(argv)
    eng = Engine(args.checkpoint, device=args.device)
    res = run(eng, build_items(args.n, [l.strip() for l in args.locales.split(",") if l.strip()]))
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "recovery_probe.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"recovery {res['recovery_rate']:.3f}  repeat {res['repeat_rate']:.3f}  "
          f"no-action {res['no_action_rate']:.3f}  n={res['n']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
