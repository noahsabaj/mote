"""Verifiable multi-turn tasks in the sim — the RLVR-1 environment (docs/shape.md § pipeline, signed 2026-08-24).

A task is a fresh world at its initial state plus a goal derived State2State-style (2608.04934): k legal
actions are explored from the initial state, the facts they changed become the goal predicates, the
exploration is pruned to the shortest sequence that still reaches them (the expert), and the world is
rebuilt for the episode. Reward = 1 iff every predicate holds at the end (the fraction is logged too).

The model acts through the tool protocol: `<|call|>sim: ivy: take key<|result|>` -> the sim applies the
action with `World.step`, renders its events in the task's locale (the same sentences as the narratives)
and the reply resumes. An action the system ignores renders as "Nothing happened."; after the step budget
(len(expert) + 2) the tool refuses. Kinship has no agent actions and is not a task domain.

Action grammar (English keys in every locale; case-insensitive; `<who>` is a person key):
  household   <who>: move to <room> | <who>: take <object> | <who>: put <object> in <container> | <who>: put down <object>
  inventory   <buyer>: buy <n> <goods> from <seller> | <who>: harvest <n> <goods>
  schedule    <who>: book <title> at <h> for <dur>h | <who>: move <title> to <h>

Seeds: training traces use 1..N (generate.py), the sim-QA probe ≥ 5 000 000; expert traces here use
≥ 2 000 000 and held-out RL tasks ≥ 6 000 000 (`EXPERT_SEED_BASE`, `HELDOUT_SEED_BASE`).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .domains import (CONTAINERS, GOODS, OBJECTS, PEOPLE, ROOMS, TITLES, WINDOWS, Calendar, Container, InRoom, Person,
                      Portable, Stock, Trace, make_trace, sample_difficulty)
from .ecs import World
from .render import (EN, JA, JA_COUNTER, LOCALES, RU_COIN_FORMS, RU_CONT, RU_GOODS_FORMS, RU_OBJ, RU_TITLES, _cap, en_n,
                     ru_n, ru_room_in)

TASK_DOMAINS = ("household", "inventory", "schedule")
EXPERT_SEED_BASE = 2_000_000
HELDOUT_SEED_BASE = 6_000_000
Predicate = Tuple[Any, ...]

TEXT = {
    "en": {"goal": "Goal: ", "instr": "Act through the sim tool, one action per call (for example ivy: take key). Say Done. once the goal holds.",
           "nothing": "Nothing happened.", "unknown": "Unknown action.", "budget": "No moves left.", "done": "Done.", "sep": " "},
    "ru": {"goal": "Цель: ", "instr": "Действуй через инструмент sim, одно действие на вызов (например ivy: take key). Скажи Done., когда цель достигнута.",
           "nothing": "Ничего не произошло.", "unknown": "Неизвестное действие.", "budget": "Ходов больше нет.", "done": "Done.", "sep": " "},
    "ja": {"goal": "目標：", "instr": "simツールで1回の呼び出しにつき1つの行動をとる（例: ivy: take key）。目標を達成したらDone.と言う。",
           "nothing": "何も起こらなかった。", "unknown": "不明な行動。", "budget": "もう動けない。", "done": "Done.", "sep": ""},
}


# --- world access ------------------------------------------------------------------------------------
def people_of(w: World) -> List[str]:
    """Household people carry Person; inventory and schedule people carry only Stock / Calendar."""
    return sorted(w.names[e] for e in w.names if any(w.get(e, C) is not None for C in (Person, Stock, Calendar)))


def fresh_world(domain: str, seed: int) -> Trace:
    """The domain's world at its initial state (the scripted trace with zero ticks)."""
    diff = {**sample_difficulty(random.Random(seed ^ 0x5EED)), "ticks": 0}
    return make_trace(domain, seed, diff)


def init_of(tr: Trace) -> Dict[str, Any]:
    """The init event's data (household / inventory); the schedule domain starts with empty calendars."""
    return tr.events[0].data if tr.events else {}


def _join_names(names: Sequence[str], locale: str) -> str:
    caps = [_cap(n) for n in names]
    if locale == "ja":
        return "、".join(caps)
    conj = " и " if locale == "ru" else " and "
    return caps[0] if len(caps) == 1 else ", ".join(caps[:-1]) + conj + caps[-1]


def initial_narrative(tr: Trace, locale: str) -> str:
    if tr.events:
        return LOCALES[locale]["event"](tr.events[0])
    who = _join_names(people_of(tr.world), locale)
    return {"en": f"{who} have no bookings yet.", "ru": f"У {who} пока нет встреч.", "ja": f"{who}にはまだ予定がない。"}[locale]


def _rooms_in_use(w: World) -> List[str]:
    return sorted({w.get(e, InRoom).room for e in w.names if w.get(e, InRoom) is not None})


def legal_actions(domain: str, w: World, init: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every action the domain's system would apply from this state (the explorer samples from these)."""
    out: List[Dict[str, Any]] = []
    if domain == "household":
        rooms = _rooms_in_use(w)
        for p in people_of(w):
            pe = w.eid(p)
            proom = w.get(pe, InRoom).room
            out += [{"kind": "move", "who": p, "to": r} for r in rooms if r != proom]
            for e, name in sorted(w.names.items(), key=lambda kv: kv[1]):
                if w.get(e, Portable) is None:
                    continue
                if w.one(e, "held_by") == pe:
                    out.append({"kind": "put_down", "who": p, "obj": name})
                    out += [{"kind": "put_in", "who": p, "obj": name, "cont": w.names[c]} for c in w.names
                            if w.get(c, Container) is not None and w.get(c, InRoom).room == proom]
                elif w.one(e, "held_by") is None and w.one(e, "inside") is None and w.get(e, InRoom).room == proom:
                    out.append({"kind": "take", "who": p, "obj": name})
    elif domain == "inventory":
        price = init["price"]
        ppl = people_of(w)
        for buyer in ppl:
            bs = w.get(w.eid(buyer), Stock)
            for seller in ppl:
                if seller == buyer:
                    continue
                ss = w.get(w.eid(seller), Stock)
                for g in sorted(price):
                    for n in (1, 2, 3):
                        if ss.goods.get(g, 0) >= n and bs.coins >= n * price[g]:
                            out.append({"kind": "trade", "buyer": buyer, "seller": seller, "goods": g, "n": n, "cost": n * price[g]})
            for g in sorted(price):
                for n in (1, 2):
                    out.append({"kind": "harvest", "who": buyer, "goods": g, "n": n, "v": 0})
    elif domain == "schedule":
        for p in people_of(w):
            cal = w.get(w.eid(p), Calendar)

            def clashes(start, end, skip=None):
                return any(start < e2 and s2 < end for j, (s2, e2, _) in enumerate(cal.slots) if j != skip)

            used = {t for _s, _e, t in cal.slots}
            for title in TITLES:
                if title in used:
                    continue
                for dur in (1, 2):
                    out += [{"kind": "book", "who": p, "title": title, "start_h": h, "dur": dur}
                            for h in WINDOWS[title] if h + dur <= 17 and not clashes(h, h + dur)]
            for i, (s, e, title) in enumerate(cal.slots):
                out += [{"kind": "move", "who": p, "i": i, "to_h": h} for h in WINDOWS[title]
                        if h != s and h + (e - s) <= 17 and not clashes(h, h + (e - s), skip=i)]
    return out


# --- the action language --------------------------------------------------------------------------------
_HOUSE = [(re.compile(r"^(\w+):\s*move to (\w+)$"), "move"), (re.compile(r"^(\w+):\s*take (\w+)$"), "take"),
          (re.compile(r"^(\w+):\s*put (\w+) in(?:to)? (\w+)$"), "put_in"), (re.compile(r"^(\w+):\s*put down (\w+)$"), "put_down")]
_INV = [(re.compile(r"^(\w+):\s*buy (\d+) (\w+) from (\w+)$"), "trade"), (re.compile(r"^(\w+):\s*harvest (\d+) (\w+)$"), "harvest")]
_SCHED = [(re.compile(r"^(\w+):\s*book (\w+) at (\d+)(?::00)? for (\d+)\s*h?$"), "book"),
          (re.compile(r"^(\w+):\s*move (\w+) to (\d+)(?::00)?$"), "move")]


def action_text(a: Dict[str, Any], cal_titles: Optional[Callable[[str, int], str]] = None) -> str:
    """The canonical text of a domain action (what the expert writes)."""
    k = a["kind"]
    if k == "move" and "to" in a:
        return f"{a['who']}: move to {a['to']}"
    if k == "take":
        return f"{a['who']}: take {a['obj']}"
    if k == "put_in":
        return f"{a['who']}: put {a['obj']} in {a['cont']}"
    if k == "put_down":
        return f"{a['who']}: put down {a['obj']}"
    if k == "trade":
        return f"{a['buyer']}: buy {a['n']} {a['goods']} from {a['seller']}"
    if k == "harvest":
        return f"{a['who']}: harvest {a['n']} {a['goods']}"
    if k == "book":
        return f"{a['who']}: book {a['title']} at {a['start_h']} for {a['dur']}h"
    if k == "move":  # schedule move by slot index -> by title
        assert cal_titles is not None
        return f"{a['who']}: move {cal_titles(a['who'], a['i'])} to {a['to_h']}"
    raise KeyError(k)


def parse_action(domain: str, text: str, w: World, init: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Text -> the domain's action dict, or None when it does not parse / names unknown things."""
    t = text.strip().lower().rstrip(".")
    ppl = set(people_of(w))
    if domain == "household":
        for rx, kind in _HOUSE:
            m = rx.match(t)
            if not m:
                continue
            g = m.groups()
            if g[0] not in ppl:
                return None
            if kind == "move":
                return {"kind": "move", "who": g[0], "to": g[1]} if g[1] in ROOMS else None
            if kind == "take":
                return {"kind": "take", "who": g[0], "obj": g[1]} if g[1] in OBJECTS and g[1] in w.names.values() else None
            if kind == "put_in":
                ok = g[1] in OBJECTS and g[2] in CONTAINERS and g[1] in w.names.values() and g[2] in w.names.values()
                return {"kind": "put_in", "who": g[0], "obj": g[1], "cont": g[2]} if ok else None
            if kind == "put_down":
                return {"kind": "put_down", "who": g[0], "obj": g[1]} if g[1] in OBJECTS and g[1] in w.names.values() else None
    elif domain == "inventory":
        price = init["price"]
        for rx, kind in _INV:
            m = rx.match(t)
            if not m:
                continue
            g = m.groups()
            if g[0] not in ppl:
                return None
            if kind == "trade":
                if g[2] not in price or g[3] not in ppl or g[3] == g[0]:
                    return None
                n = int(g[1])
                return {"kind": "trade", "buyer": g[0], "seller": g[3], "goods": g[2], "n": n, "cost": n * price[g[2]]} if 1 <= n <= 3 else None
            if kind == "harvest":
                n = int(g[1])
                return {"kind": "harvest", "who": g[0], "goods": g[2], "n": n, "v": 0} if g[2] in price and 1 <= n <= 2 else None
    elif domain == "schedule":
        for rx, kind in _SCHED:
            m = rx.match(t)
            if not m:
                continue
            g = m.groups()
            if g[0] not in ppl or g[1] not in TITLES:
                return None
            cal = w.get(w.eid(g[0]), Calendar)

            def clashes(start, end, skip=None):
                return any(start < e2 and s2 < end for j, (s2, e2, _) in enumerate(cal.slots) if j != skip)

            if kind == "book":
                h, dur = int(g[2]), int(g[3])
                if g[1] in {t2 for _s, _e, t2 in cal.slots} or dur not in (1, 2) or h not in WINDOWS[g[1]] or h + dur > 17 or clashes(h, h + dur):
                    return None  # the system trusts its caller; illegal bookings are refused here instead
                return {"kind": "book", "who": g[0], "title": g[1], "start_h": h, "dur": dur}
            if kind == "move":
                idx = next((i for i, (_s, _e, t2) in enumerate(cal.slots) if t2 == g[1]), None)
                if idx is None:
                    return None
                s, e, _ = cal.slots[idx]
                h = int(g[2])
                if h == s or h not in WINDOWS[g[1]] or h + (e - s) > 17 or clashes(h, h + (e - s), skip=idx):
                    return None
                return {"kind": "move", "who": g[0], "i": idx, "to_h": h}
    return None


# --- goals ----------------------------------------------------------------------------------------------
def _obj_fact(w: World, o: str) -> Predicate:
    e = w.eid(o)
    holder, cont = w.one(e, "held_by"), w.one(e, "inside")
    if holder is not None:
        return ("obj_held", o, w.names[holder])
    if cont is not None:
        return ("obj_in", o, w.names[cont])
    return ("obj_room", o, w.get(e, InRoom).room)


def facts_touched(domain: str, w: World, events: Sequence[Any]) -> List[Predicate]:
    """The current facts about everything the applied actions changed (the goal of a State2State task)."""
    seen: Dict[Tuple, Predicate] = {}
    for ev in events:
        d = ev.data
        if domain == "household":
            if ev.kind == "move":
                seen[("person", d["who"])] = ("person_room", d["who"], w.get(w.eid(d["who"]), InRoom).room)
                for e in w.reverse(w.eid(d["who"]), "held_by"):
                    seen[("obj", w.names[e])] = _obj_fact(w, w.names[e])
            elif "obj" in d:
                seen[("obj", d["obj"])] = _obj_fact(w, d["obj"])
        elif domain == "inventory":
            for who in ((d["buyer"], d["seller"]) if ev.kind == "trade" else (d["who"],)):
                st = w.get(w.eid(who), Stock)
                seen[("goods", who, d["goods"])] = ("goods", who, d["goods"], st.goods[d["goods"]])
                if ev.kind == "trade":
                    seen[("coins", who)] = ("coins", who, st.coins)
        elif domain == "schedule":
            cal = w.get(w.eid(d["who"]), Calendar)
            slot = next((s for s in cal.slots if s[2] == d["title"]), None)
            if slot is not None:
                seen[("booked", d["who"], d["title"])] = ("booked", d["who"], d["title"], slot[0], slot[1])
    return [seen[k] for k in sorted(seen, key=str)]


def holds(w: World, p: Predicate) -> bool:
    k = p[0]
    if k in ("obj_held", "obj_in", "obj_room"):
        return _obj_fact(w, p[1]) == p
    if k == "person_room":
        return w.get(w.eid(p[1]), InRoom).room == p[2]
    if k == "goods":
        return w.get(w.eid(p[1]), Stock).goods.get(p[2], 0) == p[3]
    if k == "coins":
        return w.get(w.eid(p[1]), Stock).coins == p[2]
    if k == "booked":
        return (p[3], p[4], p[2]) in w.get(w.eid(p[1]), Calendar).slots
    raise KeyError(k)


def score(w: World, goal: Sequence[Predicate]) -> Tuple[bool, float]:
    ok = [holds(w, p) for p in goal]
    return all(ok), (sum(ok) / len(ok) if ok else 0.0)


def _pred_text(p: Predicate, locale: str) -> str:
    k = p[0]
    if locale == "ru":
        if k == "obj_room":
            return f"{_cap(RU_OBJ[p[1]][0])} {ru_room_in(p[2])}"
        if k == "obj_in":
            return f"{_cap(RU_OBJ[p[1]][0])} в {RU_CONT[p[2]][2]}"
        if k == "obj_held":
            return f"{_cap(p[2])} держит {RU_OBJ[p[1]][1]}"
        if k == "person_room":
            return f"{_cap(p[1])} {ru_room_in(p[2])}"
        if k == "goods":
            return f"У {_cap(p[1])} {ru_n(p[3], RU_GOODS_FORMS[p[2]])}"
        if k == "coins":
            return f"У {_cap(p[1])} {ru_n(p[2], RU_COIN_FORMS)}"
        if k == "booked":
            return f"У {_cap(p[1])} «{RU_TITLES[p[2]]}» с {p[3]}:00 до {p[4]}:00"
    if locale == "ja":
        if k == "obj_room":
            return f"{JA['objects'][p[1]]}は{JA['rooms'][p[2]]}にある"
        if k == "obj_in":
            return f"{JA['objects'][p[1]]}は{JA['containers'][p[2]]}の中にある"
        if k == "obj_held":
            return f"{_cap(p[2])}は{JA['objects'][p[1]]}を持っている"
        if k == "person_room":
            return f"{_cap(p[1])}は{JA['rooms'][p[2]]}にいる"
        if k == "goods":
            return f"{_cap(p[1])}は{JA['goods'][p[2]]}を{p[3]}{JA_COUNTER[p[2]]}持っている"
        if k == "coins":
            return f"{_cap(p[1])}はコインを{p[2]}枚持っている"
        if k == "booked":
            return f"{_cap(p[1])}は{p[3]}時から{p[4]}時まで{JA['titles'][p[2]]}がある"
    if k == "obj_room":
        return f"{_cap(EN['objects'][p[1]])} is in {EN['rooms'][p[2]]}"
    if k == "obj_in":
        return f"{_cap(EN['objects'][p[1]])} is in {EN['containers'][p[2]]}"
    if k == "obj_held":
        return f"{_cap(p[2])} has {EN['objects'][p[1]]}"
    if k == "person_room":
        return f"{_cap(p[1])} is in {EN['rooms'][p[2]]}"
    if k == "goods":
        return f"{_cap(p[1])} has {en_n(p[3], p[2])}"
    if k == "coins":
        return f"{_cap(p[1])} has {p[2]} coin{'s' if p[2] != 1 else ''}"
    if k == "booked":
        return f"{_cap(p[1])} has {EN['titles'][p[2]]} from {p[3]}:00 to {p[4]}:00"
    raise KeyError(k)


def goal_text(goal: Sequence[Predicate], locale: str) -> str:
    end = "。" if locale == "ja" else "."
    return TEXT[locale]["goal"] + end.join(_pred_text(p, locale) for p in goal) + end


# --- tasks ------------------------------------------------------------------------------------------------
@dataclass
class Task:
    domain: str
    seed: int
    locale: str
    goal: List[Predicate]
    narrative: str
    expert: List[str]  # canonical action texts, pruned
    budget: int
    k_explored: int
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return f"{self.narrative}\n\n{goal_text(self.goal, self.locale)}\n{TEXT[self.locale]['instr']}"


def _cal_titles(w: World) -> Callable[[str, int], str]:
    return lambda who, i: w.get(w.eid(who), Calendar).slots[i][2]


def _apply(domain: str, seed: int, texts: Sequence[str]) -> Tuple[Trace, List[Any]]:
    """Fresh world + the given action texts applied in order; returns (trace, events)."""
    tr = fresh_world(domain, seed)
    init = init_of(tr)
    evs: List[Any] = []
    for t in texts:
        a = parse_action(domain, t, tr.world, init)
        if a is not None:
            evs += tr.world.step([a])
    return tr, evs


def make_task(domain: str, seed: int, locale: str = "en", k: Optional[int] = None) -> Task:
    """Explore k legal actions from the initial state, take the changed facts as the goal, prune the
    exploration to the shortest sequence that still reaches them (the expert), rebuild for the episode."""
    assert domain in TASK_DOMAINS, domain
    rng = random.Random(seed ^ 0xA5C7)
    k = k or rng.randint(1, 5)
    tr = fresh_world(domain, seed)
    try:
        init = init_of(tr)
        narrative = initial_narrative(tr, locale)
        texts: List[str] = []
        evs: List[Any] = []
        for _ in range(k):
            legal = legal_actions(domain, tr.world, init)
            if not legal:
                break
            a = rng.choice(legal)
            texts.append(action_text(a, _cal_titles(tr.world)))  # titles resolved before the step
            evs += tr.world.step([a])
        goal = facts_touched(domain, tr.world, evs)
    finally:
        tr.world.close()
    t0 = fresh_world(domain, seed)  # a fact the exploration circled back to already holds: not a goal
    goal = [p for p in goal if not holds(t0.world, p)]
    t0.world.close()
    if not goal:  # nothing changed in the end (or every action was a no-op): take the next world
        return make_task(domain, seed + 1_000_003, locale, k)
    # greedy pruning: drop any action whose removal still reaches the goal
    expert = list(texts)
    i = 0
    while i < len(expert):
        trial = expert[:i] + expert[i + 1:]
        t2, _ = _apply(domain, seed, trial)
        ok = score(t2.world, goal)[0]
        t2.world.close()
        if ok:
            expert = trial
        else:
            i += 1
    return Task(domain, seed, locale, goal, narrative, expert, len(expert) + 2, k,
                meta={"n_goal": len(goal), "k_explored": k, "n_expert": len(expert)})


def heldout_tasks(n: int, locales: Sequence[str] = ("en", "ru", "ja"), seed_base: int = HELDOUT_SEED_BASE) -> List[Task]:
    out: List[Task] = []
    seed = seed_base
    while len(out) < n:
        seed += 1
        out.append(make_task(TASK_DOMAINS[seed % len(TASK_DOMAINS)], seed, locales[len(out) % len(locales)]))
    return out


# --- the environment / tool -------------------------------------------------------------------------------
class SimEnv:
    """One episode: the task's world, the step budget, the tool function the engine calls."""

    def __init__(self, task: Task):
        self.task = task
        self.trace = fresh_world(task.domain, task.seed)
        self.init = init_of(self.trace)
        self.steps = 0
        self.log: List[Dict[str, Any]] = []

    @property
    def world(self) -> World:
        return self.trace.world

    def act(self, text: str) -> str:
        """The `sim` tool: apply one action text, return the rendered events (the observation)."""
        T = TEXT[self.task.locale]
        if self.steps >= self.task.budget:
            self.log.append({"text": text, "result": "budget"})
            return T["budget"]
        self.steps += 1
        a = parse_action(self.task.domain, text, self.world, self.init)
        if a is None:
            self.log.append({"text": text, "result": "unknown"})
            return T["unknown"]
        evs = self.world.step([a])
        self.log.append({"text": text, "result": "ok" if evs else "noop"})
        if not evs:
            return T["nothing"]
        return T["sep"].join(s for s in (LOCALES[self.task.locale]["event"](e) for e in evs) if s)

    def score(self) -> Tuple[bool, float]:
        return score(self.world, self.task.goal)

    def close(self) -> None:
        self.world.close()


def expert_messages(task: Task) -> List[Dict[str, Any]]:
    """The expert trajectory as chat messages with tool parts (SFT traces; `build_local` renders them)."""
    env = SimEnv(task)
    try:
        parts: List[Dict[str, str]] = []
        for t in task.expert:
            parts.append({"type": "call", "text": f"sim: {t}"})
            parts.append({"type": "result", "text": env.act(t)})
        assert env.score()[0], (task.domain, task.seed, task.expert)
        parts.append({"type": "text", "text": TEXT[task.locale]["done"]})
    finally:
        env.close()
    return [{"role": "user", "content": task.prompt}, {"role": "assistant", "content": "", "parts": parts}]


def main(argv=None):
    """Write expert traces as chat JSONL for `build_local --chat ... --sft`:
        python -m mote.sim.tasks --out data/sim_traces --n 20000
    """
    import argparse
    import json
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="prefix; writes <out>.jsonl and <out>.stats.json")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed-base", type=int, default=EXPERT_SEED_BASE)
    ap.add_argument("--locales", default="en,ru,ja")
    args = ap.parse_args(argv)
    locales = [l.strip() for l in args.locales.split(",") if l.strip()]
    stats: Counter = Counter()
    seed = args.seed_base
    n = 0
    with open(f"{args.out}.jsonl", "w", encoding="utf-8") as f:
        while n < args.n:
            seed += 1
            domain = TASK_DOMAINS[seed % len(TASK_DOMAINS)]
            locale = locales[n % len(locales)]
            task = make_task(domain, seed, locale)
            msgs = expert_messages(task)
            f.write(json.dumps({"messages": msgs, "meta": {"domain": domain, "locale": locale, "seed": seed, **task.meta}}, ensure_ascii=False) + "\n")
            stats[f"domain:{domain}"] += 1
            stats[f"n_expert:{len(task.expert)}"] += 1
            stats[f"n_goal:{len(task.goal)}"] += 1
            n += 1
    with open(f"{args.out}.stats.json", "w") as f:
        json.dump(dict(sorted(stats.items())), f, indent=1)
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
