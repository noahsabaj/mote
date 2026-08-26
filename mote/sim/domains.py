"""The four micro-domains. Each builds a world, scripts random actions, and derives questions
whose answers are read off the true world state (correct by construction). Every question
carries a plausible-wrong answer for DPO: a stale historical value, a same-type wrong entity,
an off-by-one, a yes/no flip, or the current state when the question asks about the start.

Every domain's dynamics is a system on the world (`World.step(actions)`); the script only
*chooses* actions from the true state + rng. Relationships (held_by, inside, parent_of,
spouse_of) are first-class on the world instead of string fields. The scripting order of rng
calls is the dataset's identity — the esper port (2026-08-24) reproduces the gated build
byte for byte.

Difficulty axes (logged per sample): n_entities, n_ticks (events between question), distractors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ecs import Event, World

# entity display keys are locale-independent; renderers map them to words
PEOPLE = ["mara", "jon", "ivy", "tomas", "lena", "kofi", "sana", "rui"]
ROOMS = ["kitchen", "garden", "study", "hall", "cellar", "attic"]
OBJECTS = ["key", "book", "lamp", "coin", "apple", "letter", "cup", "knife"]
CONTAINERS = ["box", "basket", "drawer", "chest"]
GOODS = ["apples", "loaves", "candles", "nails", "eggs"]
KIN_NAMES = ["ada", "bruno", "cora", "dario", "elsa", "felix", "greta", "hugo", "irene", "janos", "katia", "luca"]
TITLES = ["standup", "review", "lunch", "planning", "call", "workshop"]
WINDOWS = {"standup": range(8, 11), "lunch": range(11, 14), "workshop": range(9, 16),
           "review": range(9, 17), "planning": range(9, 17), "call": range(8, 17)}


@dataclass
class Q:
    qtype: str
    args: Dict[str, Any]     # entity keys the renderer needs
    answer: Any              # structured truth (name key, number, bool, pair)
    wrong: Any               # plausible-wrong for DPO
    wrong_kind: str          # "stale" | "wrong_entity" | "off_by_one" | "flip" | "current"


@dataclass
class Trace:
    domain: str
    seed: int
    difficulty: Dict[str, int]
    world: World
    events: List[Event]
    questions: List[Q]


# ---------------------------------------------------------------- household
@dataclass
class InRoom:
    room: str


@dataclass
class Person:
    pass


@dataclass
class Container:
    pass


@dataclass
class Portable:
    pass


# --- failures ---------------------------------------------------------------------------------------
# Until 2026-08-26 the scripts only ever proposed legal actions ("the script has already checked
# feasibility"), so no action in any narrative could fail and every result was a successful restatement of
# its call. Measured over 20k traces: 70 % of result bytes were fully determined by (locale, call), and
# only 0.27 % of the trace stream was genuinely unpredictable. A world where nothing can fail has almost
# no state-dependent outcome to model, and it is a weak RL substrate besides — `mote/sim/tasks.py` refuses
# illegal actions, so RLVR-1 would have met its first refusal having never seen one.
#
# The systems below already detected illegality; they simply emitted nothing. Now they emit a `failed`
# event carrying the reason, which the renderers turn into prose and which the questions can be asked
# about. `_maybe_illegal` in each script chooses an illegal action at `p_fail`, weighted toward the ones
# whose refusal *reveals* state (taking an object someone else holds says who holds it).
FAIL_REASONS = ("not_here", "held_by_other", "already_holding", "inside_container", "not_holding",
                "no_goods", "no_coins", "slot_clash", "no_such_booking")


def _failed(w: World, kind: str, why: str, **data) -> Event:
    return Event(w.t, "failed", {"kind": kind, "why": why, **data})


def household_system(w: World, actions: List[Dict[str, Any]]) -> List[Event]:
    """move / take / put_in / put_down against InRoom + the held_by and inside relations.

    An action that cannot be applied emits a `failed` event rather than nothing, so the narrative records
    the attempt and its reason. The world state is unchanged, which is what makes the questions harder:
    the reader has to notice that the object did not move."""
    events: List[Event] = []
    for a in actions:
        who = w.eid(a["who"])
        if a["kind"] == "move":
            w.get(who, InRoom).room = a["to"]
            for o in w.reverse(who, "held_by"):  # held objects travel with the holder
                w.get(o, InRoom).room = a["to"]
            events.append(Event(w.t, "move", {"who": a["who"], "to": a["to"]}))
        elif a["kind"] == "take":
            o = w.eid(a["obj"])
            holder = w.one(o, "held_by")
            cont = w.one(o, "inside")
            if holder == who:  # self-hold: "Mara tried to take the key, but Mara was holding it" reads
                events.append(_failed(w, "take", "already_holding", who=a["who"], obj=a["obj"]))
            elif holder is not None:
                events.append(_failed(w, "take", "held_by_other", who=a["who"], obj=a["obj"],
                                      holder=w.name(holder)))
            elif cont is not None:
                events.append(_failed(w, "take", "inside_container", who=a["who"], obj=a["obj"],
                                      cont=w.name(cont)))
            elif w.get(o, InRoom).room != w.get(who, InRoom).room:
                events.append(_failed(w, "take", "not_here", who=a["who"], obj=a["obj"],
                                      room=w.get(o, InRoom).room))
            else:
                w.relate(o, "held_by", who)
                events.append(Event(w.t, "take", {"who": a["who"], "obj": a["obj"]}))
        elif a["kind"] == "put_in":
            o = w.eid(a["obj"])
            if w.one(o, "held_by") == who:
                w.unrelate(o, "held_by")
                w.relate(o, "inside", w.eid(a["cont"]))
                events.append(Event(w.t, "put_in", {"who": a["who"], "obj": a["obj"], "cont": a["cont"]}))
            else:
                events.append(_failed(w, "put_in", "not_holding", who=a["who"], obj=a["obj"], cont=a["cont"]))
        elif a["kind"] == "put_down":
            o = w.eid(a["obj"])
            if w.one(o, "held_by") == who:
                w.unrelate(o, "held_by")
                events.append(Event(w.t, "put_down", {"who": a["who"], "obj": a["obj"], "room": w.get(o, InRoom).room}))
            else:
                events.append(_failed(w, "put_down", "not_holding", who=a["who"], obj=a["obj"]))
    return events


def _divert_household(rng, action, objects, containers, byname, w, snap, who) -> Dict[str, Any]:
    """A different final action from the same state — the counterfactual branch of a minimal pair."""
    held = [o for o in objects if w.one(byname[o], "held_by") == byname[who]]
    room = w.get(byname[who], InRoom).room
    here = [o for o in objects if snap(o) == {"room": room, "held": "", "cont": ""}]
    alts = []
    if held:
        alts.append({"kind": "put_down", "who": who, "obj": held[0]})
        if containers:
            alts.append({"kind": "put_in", "who": who, "obj": held[0], "cont": containers[0]})
    if here:
        alts.append({"kind": "take", "who": who, "obj": here[-1]})
    alts = [a for a in alts if a != action]
    return alts[0] if alts else action


def _household(seed: int, diff: Dict[str, int]) -> Trace:
    rng = random.Random(seed)
    w = World(seed)
    w.add_system(household_system)
    people = rng.sample(PEOPLE, diff["people"])
    rooms = rng.sample(ROOMS, diff["rooms"])
    objects = rng.sample(OBJECTS, diff["objects"])
    containers = rng.sample(CONTAINERS, min(2, diff["objects"]))
    byname: Dict[str, int] = {}
    for p in people:
        byname[p] = w.spawn(p)
        w.add(byname[p], Person())
        w.add(byname[p], InRoom(rng.choice(rooms)))
    cont_room = {c: rng.choice(rooms) for c in containers}
    for c in containers:
        byname[c] = w.spawn(c)
        w.add(byname[c], Container())
        w.add(byname[c], InRoom(cont_room[c]))
    for o in objects:
        byname[o] = w.spawn(o)
        w.add(byname[o], Portable())
        w.add(byname[o], InRoom(rng.choice(rooms)))

    def snap(o):
        e = byname[o]
        holder, cont = w.one(e, "held_by"), w.one(e, "inside")
        return {"room": w.get(e, InRoom).room, "held": w.names[holder] if holder is not None else "",
                "cont": w.names[cont] if cont is not None else ""}

    prev_room: Dict[str, str] = {}
    history: List[Tuple[int, str, Dict[str, str], Tuple[str, Optional[str]]]] = [
        (0, o, snap(o), ("init", None)) for o in objects]
    events: List[Event] = [Event(0, "init_household", {
        "people": {p: w.get(byname[p], InRoom).room for p in people},
        "containers": dict(cont_room),
        "objects": {o: w.get(byname[o], InRoom).room for o in objects},
    })]
    p_fail = diff.get("p_fail", 0) / 100.0
    # --- counterfactual fork -------------------------------------------------------------------------
    # A minimal pair needs two narratives that differ in exactly one event and have different answers.
    # That normally wants a replay-to-step API, which does not exist (the parked PIVOT item wants the same
    # one). It is unnecessary for the LAST action: the RNG draws that choose it have already happened, so
    # overriding the chosen action leaves every earlier byte identical and changes only the final sentence
    # and the answers that depend on it. `divert` asks for that override; `make_counterfactual` below
    # builds the pair. 2605.17528 (CausalSynth) generates causal skeletons from an SCM and pays an LLM to
    # realise them — Mote's sim *is* the SCM, so the expensive half is free.
    divert = diff.get("divert")
    for tick in range(diff["ticks"]):
        # script one action per tick from the true state, then let the SYSTEM apply it
        p = rng.choice(people)
        proom = w.get(byname[p], InRoom).room
        held_by_p = [o for o in objects if w.one(byname[o], "held_by") == byname[p]]
        here = [o for o in objects if snap(o) == {"room": proom, "held": "", "cont": ""}]
        # A deliberate illegal attempt, weighted toward the ones whose refusal *reveals* state: failing to
        # take an object says who holds it or which room it is in, which is information the narrative does
        # not otherwise volunteer. A failed action leaves the world unchanged, so the questions below stay
        # correct — they just get harder, because the reader has to notice nothing moved.
        illegal = None
        if rng.random() < p_fail:
            elsewhere = [o for o in objects if snap(o)["room"] != proom or snap(o)["held"] or snap(o)["cont"]]
            not_held = [o for o in objects if w.one(byname[o], "held_by") != byname[p]]
            # Weighted, not uniform. "You are not holding that" is a bare negation; "Ivy has it" and "it is
            # in the cellar" each hand the reader a fact the narrative did not otherwise state. A uniform
            # draw put 69 % of household failures on the uninformative branch, so `take` gets the weight.
            choices, weights = [], []
            if elsewhere:
                choices.append({"kind": "take", "who": p, "obj": rng.choice(elsewhere)}); weights.append(6)
            if not_held:
                choices.append({"kind": "put_down", "who": p, "obj": rng.choice(not_held)}); weights.append(1)
                if containers:
                    choices.append({"kind": "put_in", "who": p, "obj": rng.choice(not_held), "cont": rng.choice(containers)})
                    weights.append(1)
            if choices:
                illegal = rng.choices(choices, weights)[0]
        act = rng.random()
        if illegal is not None:
            action = illegal
        elif act < 0.4 or (not here and not held_by_p):
            options = [r for r in rooms if r != proom and r != prev_room.get(p)] or [r for r in rooms if r != proom]
            action = {"kind": "move", "who": p, "to": rng.choice(options)}
            prev_room[p] = proom  # noqa: E501
        elif (act < 0.7 and here) or not held_by_p:
            action = {"kind": "take", "who": p, "obj": rng.choice(here)}
        else:
            o = rng.choice(held_by_p)
            conts_here = [c for c in containers if cont_room[c] == proom]
            if conts_here and rng.random() < 0.5:
                action = {"kind": "put_in", "who": p, "obj": o, "cont": rng.choice(conts_here)}
            else:
                action = {"kind": "put_down", "who": p, "obj": o}
        if divert and tick == diff["ticks"] - 1:
            action = _divert_household(rng, action, objects, containers, byname, w, snap, p)
        new_events = w.step([action])
        events.extend(new_events)
        for e in new_events:
            if e.kind == "failed":
                continue  # a failure changes nothing, so it is not a history entry
            # `cause` names the event that produced this state, so a retrodiction question can anchor on
            # something the narrative actually said ("before Ivy picked it up") rather than on a tick
            # index the reader never sees.
            cause = (e.kind, e.data.get("who"))
            if "obj" in e.data:
                history.append((e.t, e.data["obj"], snap(e.data["obj"]), cause))
            elif e.kind == "move":
                for o in objects:
                    if w.one(byname[o], "held_by") == byname[e.data["who"]]:
                        history.append((e.t, o, snap(o), cause))
    obj_state = {o: snap(o) for o in objects}
    loc = {p: w.get(byname[p], InRoom).room for p in people}

    qs: List[Q] = []
    rngq = random.Random(seed + 1)
    for o in rngq.sample(objects, min(len(objects), 3)):
        s = obj_state[o]
        first = next(h for h in history if h[1] == o)[2]
        touches = [h for h in history if h[1] == o]
        if s["held"]:
            qs.append(Q("who_has_obj", {"obj": o}, ("person", s["held"]),
                        ("person", rngq.choice([p for p in people if p != s["held"]])), "wrong_entity"))
            continue
        elif s["cont"]:
            ans = ("cont", (s["cont"], cont_room[s["cont"]]))
            wrong, wk = ("room", first["room"]), "stale"
        else:
            ans = ("room", s["room"])
            visited = {h[2]["room"] for h in history if h[1] == o} - {s["room"]}
            if visited:
                wrong, wk = ("room", sorted(visited)[0]), "stale"
            else:
                wrong, wk = ("room", rngq.choice([r for r in rooms if r != s["room"]])), "wrong_entity"
        qs.append(Q("where_obj", {"obj": o}, ans, wrong, wk))
        if first["room"] != s["room"] or s["cont"]:
            now = ("cont", (s["cont"], cont_room[s["cont"]])) if s["cont"] else ("room", s["room"])
            qs.append(Q("where_obj_start", {"obj": o}, ("room", first["room"]), now, "current"))
        # Retrodiction at an arbitrary point, not just the start. Every question in this generator asked
        # "where is X *now*"; left-to-right training only ever exercises the forward direction, which is
        # the blind spot FIM (2607.12463) found in position and this is the same one in time. The history
        # already held the answer -- only `where_obj_start` (2.8 % of questions) ever read it.
        if len(touches) >= 3:
            k = rngq.randrange(1, len(touches) - 1)
            before, after = touches[k - 1][2], touches[k]
            kind, actor = after[3]
            if actor and before["room"] and not before["held"] and not before["cont"] and kind != "init":
                qs.append(Q("where_obj_before", {"obj": o, "verb": kind, "who": actor},
                            ("room", before["room"]),
                            ("room", s["room"]) if s["room"] != before["room"] else
                            ("room", rngq.choice([r for r in rooms if r != before["room"]])),
                            "current" if s["room"] != before["room"] else "wrong_entity"))
    p = rngq.choice(people)
    n_here = sum(1 for o, s in obj_state.items() if s["room"] == loc[p] and not s["held"] and not s["cont"])
    qs.append(Q("where_person", {"who": p}, ("room", loc[p]),
                ("room", rngq.choice([r for r in rooms if r != loc[p]])), "wrong_entity"))
    qs.append(Q("count_loose_in_room", {"room": loc[p]}, ("num", n_here), ("num", n_here + rngq.choice([-1, 1]) if n_here > 0 else n_here + 1), "off_by_one"))
    return Trace("household", seed, diff, w, events, qs)


# ---------------------------------------------------------------- inventory / trade
@dataclass
class Stock:
    goods: Dict[str, int] = field(default_factory=dict)
    coins: int = 0


def inventory_system(w: World, actions: List[Dict[str, Any]]) -> List[Event]:
    """trade / harvest against Stock. A trade the seller or buyer cannot cover emits `failed`."""
    events: List[Event] = []
    for a in actions:
        if a["kind"] == "trade":
            buyer, seller = w.get(w.eid(a["buyer"]), Stock), w.get(w.eid(a["seller"]), Stock)
            g, n, cost = a["goods"], a["n"], a["cost"]
            if seller.goods[g] < n:
                events.append(_failed(w, "trade", "no_goods", buyer=a["buyer"], seller=a["seller"],
                                      goods=g, n=n, have=seller.goods[g]))
            elif buyer.coins < cost:
                events.append(_failed(w, "trade", "no_coins", buyer=a["buyer"], seller=a["seller"],
                                      goods=g, n=n, cost=cost, have=buyer.coins))
            else:
                seller.goods[g] -= n
                buyer.goods[g] += n
                buyer.coins -= cost
                seller.coins += cost
                events.append(Event(w.t, "trade", {"buyer": a["buyer"], "seller": a["seller"], "goods": g, "n": n, "cost": cost}))
        elif a["kind"] == "harvest":
            w.get(w.eid(a["who"]), Stock).goods[a["goods"]] += a["n"]
            events.append(Event(w.t, "harvest", {"who": a["who"], "goods": a["goods"], "n": a["n"], "v": a["v"]}))
    return events


def _inventory(seed: int, diff: Dict[str, int]) -> Trace:
    rng = random.Random(seed)
    w = World(seed)
    w.add_system(inventory_system)
    people = rng.sample(PEOPLE, diff["people"])
    goods = rng.sample(GOODS, min(diff["objects"], len(GOODS)))
    price = {g: rng.randint(1, 5) for g in goods}
    stock: Dict[str, Stock] = {}
    for p in people:
        eid = w.spawn(p)
        st = Stock({g: rng.randint(0, 6) for g in goods}, rng.randint(10, 30))
        stock[p] = st
        w.add(eid, st)
    start = {p: (dict(s.goods), s.coins) for p, s in stock.items()}
    events: List[Event] = [Event(0, "init_stock", {"start": {p: {"goods": g, "coins": c} for p, (g, c) in start.items()},
                                                    "price": dict(price)})]
    p_fail = diff.get("p_fail", 0) / 100.0
    # Per-tick stock snapshots. Only the start and the end were ever recoverable before 2026-08-26, so
    # every count question was "how many now"; this is the same retrodiction the household domain gained.
    hist: List[Tuple[int, Dict[str, Tuple[Dict[str, int], int]], Tuple[str, str, str]]] = []
    for t in range(diff["ticks"]):
        buyer, seller = rng.sample(people, 2)
        g = rng.choice(goods)
        n = rng.randint(1, 3)
        # A trade nobody can cover. Its refusal is the most informative failure in this domain: it says
        # what the seller actually has, or what the buyer can actually afford — numbers the narrative
        # otherwise only reveals through the running arithmetic.
        if rng.random() < p_fail:
            if rng.random() < 0.5:                       # seller cannot supply: reveals what they have
                over = stock[seller].goods[g] + rng.randint(1, 3)
                action = {"kind": "trade", "buyer": buyer, "seller": seller, "goods": g, "n": over,
                          "cost": over * price[g]}
            else:                                        # buyer cannot pay: reveals their purse
                have = stock[seller].goods[g]
                n2 = max(1, have)
                action = {"kind": "trade", "buyer": buyer, "seller": seller, "goods": g, "n": n2,
                          "cost": stock[buyer].coins + rng.randint(1, 9)}
            events.extend(w.step([action]))
            continue
        if stock[seller].goods[g] >= n and stock[buyer].coins >= n * price[g]:
            action = {"kind": "trade", "buyer": buyer, "seller": seller, "goods": g, "n": n, "cost": n * price[g]}
        else:
            g2 = rng.choice(goods)
            n2 = rng.randint(1, 2)
            action = {"kind": "harvest", "who": buyer, "goods": g2, "n": n2, "v": t % 3}
        snap_before = {q: (dict(s.goods), s.coins) for q, s in stock.items()}
        new = w.step([action])
        events.extend(new)
        for e in new:
            if e.kind == "trade":
                hist.append((e.t, snap_before, ("trade", e.data["buyer"], e.data["seller"])))
            elif e.kind == "harvest":
                hist.append((e.t, snap_before, ("harvest", e.data["who"], e.data["goods"])))
    qs: List[Q] = []
    rngq = random.Random(seed + 1)
    # Retrodiction: the count *before* a named exchange, anchored on an event the narrative described.
    for h_t, before, cause in ([hist[len(hist) // 2]] if len(hist) >= 3 else []):
        if cause[0] == "trade":
            who = cause[1]
            g = rngq.choice(goods)
            was, now = before[who][0][g], stock[who].goods[g]
            if was != now:
                qs.append(Q("count_goods_before", {"who": who, "goods": g, "other": cause[2]},
                            ("num", was), ("num", now), "current"))
    for _ in range(3):
        p = rngq.choice(people)
        g = rngq.choice(goods)
        n = stock[p].goods[g]
        wrong = start[p][0][g] if start[p][0][g] != n else n + rngq.choice([1, 2])
        qs.append(Q("count_goods", {"who": p, "goods": g}, ("num", n), ("num", wrong),
                    "stale" if start[p][0][g] != n else "off_by_one"))
    p = rngq.choice(people)
    qs.append(Q("count_coins", {"who": p}, ("num", stock[p].coins),
                ("num", start[p][1] if start[p][1] != stock[p].coins else stock[p].coins + 2),
                "stale" if start[p][1] != stock[p].coins else "off_by_one"))
    a, b = rngq.sample(people, 2)
    g = rngq.choice(goods)
    more = a if stock[a].goods[g] >= stock[b].goods[g] else b
    qs.append(Q("who_has_more", {"a": a, "b": b, "goods": g}, ("person", more),
                ("person", b if more == a else a), "wrong_entity"))
    return Trace("inventory", seed, diff, w, events, qs)


# ---------------------------------------------------------------- kinship (static relations)
def _kinship(seed: int, diff: Dict[str, int]) -> Trace:
    rng = random.Random(seed)
    w = World(seed)
    n = max(6, diff["people"] + 4)
    names = rng.sample(KIN_NAMES, n)
    # three generations: g0 couple -> g1 children (married in from outside) -> g2 children
    g0 = names[:2]
    g1 = names[2 : 2 + max(2, n // 3)]
    g2 = names[2 + max(2, n // 3) :]
    parents: Dict[str, Tuple[str, str]] = {}
    spouses: Dict[str, str] = {g0[0]: g0[1], g0[1]: g0[0]}
    for c in g1:
        parents[c] = (g0[0], g0[1])
    pairs = []
    inlaws = g2[: len(g1) // 2]  # marry IN from outside the bloodline, never a sibling
    g2 = g2[len(inlaws):]
    for a, b in zip(g1, inlaws):
        spouses[a], spouses[b] = b, a
        pairs.append((a, b))
    for j, c in enumerate(g2):
        if not pairs:
            break
        a, b = pairs[j % len(pairs)]
        parents[c] = (a, b)
    byname = {name: w.spawn(name) for name in names}
    for name in names:
        w.add(byname[name], Person())
    for c, (p1, p2) in parents.items():
        w.relate(byname[p1], "parent_of", byname[c])
        w.relate(byname[p2], "parent_of", byname[c])
    for a, b in spouses.items():
        w.relate(byname[a], "spouse_of", byname[b])
    events = [Event(0, "family", {"parents": parents, "spouses": spouses, "gens": [g0, g1, g2]})]
    qs: List[Q] = []
    rngq = random.Random(seed + 1)
    g2_with_parents = [c for c in g2 if c in parents]
    if g2_with_parents:
        c = rngq.choice(g2_with_parents)
        mother = parents[c][0]
        gps = parents.get(mother)
        if gps:
            decoy = rngq.choice([x for x in names if x not in gps and x != mother and x != c])
            qs.append(Q("grandparent", {"who": c, "via": mother}, ("persons", tuple(gps)),
                        ("persons", (gps[0], decoy)), "wrong_entity"))
        sibs = [x for x in g2_with_parents if x != c and parents[x] == parents[c]]
        qs.append(Q("count_siblings", {"who": c}, ("num", len(sibs)),
                    ("num", len(sibs) + rngq.choice([1, -1]) if sibs else 1), "off_by_one"))
    p = rngq.choice(g1)
    kids = sorted(w.names[k] for k in w.related(byname[p], "parent_of"))
    qs.append(Q("count_children", {"who": p}, ("num", len(kids)),
                ("num", len(kids) + rngq.choice([1, 2]) if not kids else len(kids) + rngq.choice([1, -1])), "off_by_one"))
    if spouses.get(p):
        qs.append(Q("spouse_of", {"who": p}, ("person", spouses[p]),
                    ("person", rngq.choice([x for x in names if x not in (p, spouses[p])])), "wrong_entity"))
    return Trace("kinship", seed, diff, w, events, qs)


# ---------------------------------------------------------------- schedule
@dataclass
class Calendar:
    slots: List[Tuple[int, int, str]] = field(default_factory=list)  # (start_h, end_h, title)


def schedule_system(w: World, actions: List[Dict[str, Any]]) -> List[Event]:
    """book / move against Calendar. Unlike the other two this had no feasibility check at all — the
    script guaranteed a clash-free slot — so the clash test lives here now and `failed` is emitted for a
    double-booking or a move of a booking that does not exist."""
    events: List[Event] = []
    for a in actions:
        cal = w.get(w.eid(a["who"]), Calendar)

        def clashes(start, end, skip=None):
            return any(start < e2 and s2 < end for j, (s2, e2, _t) in enumerate(cal.slots) if j != skip)

        if a["kind"] == "move":
            if not (0 <= a["i"] < len(cal.slots)):
                events.append(_failed(w, "move", "no_such_booking", who=a["who"], i=a["i"]))
                continue
            s, e, title = cal.slots[a["i"]]
            ns = a["to_h"]
            if clashes(ns, ns + (e - s), skip=a["i"]):
                events.append(_failed(w, "move", "slot_clash", who=a["who"], title=title, to_h=ns))
                continue
            cal.slots[a["i"]] = (ns, ns + (e - s), title)
            events.append(Event(w.t, "moved", {"who": a["who"], "title": title, "from_h": s, "to_h": ns, "end_h": ns + (e - s)}))
        elif a["kind"] == "book":
            if clashes(a["start_h"], a["start_h"] + a["dur"]):
                events.append(_failed(w, "book", "slot_clash", who=a["who"], title=a["title"],
                                      start_h=a["start_h"], end_h=a["start_h"] + a["dur"]))
                continue
            cal.slots.append((a["start_h"], a["start_h"] + a["dur"], a["title"]))
            events.append(Event(w.t, "booked", {"who": a["who"], "title": a["title"], "start_h": a["start_h"], "end_h": a["start_h"] + a["dur"]}))
    return events


def _schedule(seed: int, diff: Dict[str, int]) -> Trace:
    rng = random.Random(seed)
    w = World(seed)
    w.add_system(schedule_system)
    people = rng.sample(PEOPLE, diff["people"])
    cal: Dict[str, Calendar] = {}
    for p in people:
        eid = w.spawn(p)
        m = Calendar()
        cal[p] = m
        w.add(eid, m)
    events: List[Event] = []
    p_fail = diff.get("p_fail", 0) / 100.0
    for _ in range(diff["ticks"]):
        p = rng.choice(people)

        def clashes(start, end, skip=None):
            return any(start < e2 and s2 < end for j, (s2, e2, _) in enumerate(cal[p].slots) if j != skip)

        # Deliberately book over an existing slot. `tasks.py` refuses exactly this at RL time, so before
        # 2026-08-26 the model would have met its first double-booking refusal during RLVR-1 having never
        # seen one in training — and it would have been the uninformative "Unknown action." at that, since
        # the parser rejected clashes before they could reach this check.
        if rng.random() < p_fail:
            # Two ways to be refused, and both are reachable from tasks.py at RL time: double-booking, and
            # moving a booking that does not exist. The second is what a model does when it hallucinates a
            # slot, so the corpus has to contain the refusal it will get.
            if cal[p].slots and rng.random() < 0.75:
                s, e, _title = cal[p].slots[rng.randrange(len(cal[p].slots))]
                free_titles = [x for x in TITLES if x not in {t2 for _s, _e, t2 in cal[p].slots}]
                if free_titles:
                    events.extend(w.step([{"kind": "book", "who": p, "title": rng.choice(free_titles),
                                           "start_h": s, "dur": max(1, e - s)}]))
                    continue
            else:
                events.extend(w.step([{"kind": "move", "who": p, "i": len(cal[p].slots) + rng.randint(0, 2),
                                       "to_h": rng.randint(8, 16)}]))
                continue

        if cal[p].slots and rng.random() < 0.3:  # move a booking (keeps its length, never self-overlaps)
            i = rng.randrange(len(cal[p].slots))
            s, e, title = cal[p].slots[i]
            cands = [h for h in range(8, 17) if h != s and h in WINDOWS[title] and not clashes(h, h + (e - s), skip=i)]
            if not cands:
                continue
            action = {"kind": "move", "who": p, "i": i, "to_h": rng.choice(cands)}
        else:
            free_titles = [x for x in TITLES if x not in {t2 for _s, _e, t2 in cal[p].slots}]
            if not free_titles:
                continue  # a repeated title would make a later "moved X" ambiguous
            dur = rng.choice([1, 2])
            cands = [h for h in range(8, 17) if not clashes(h, h + dur)]
            if not cands:
                continue
            title = rng.choice(free_titles)
            cands = [h for h in cands if h in WINDOWS[title]] or cands
            action = {"kind": "book", "who": p, "title": title, "start_h": rng.choice(cands), "dur": dur}
        events.extend(w.step([action]))
    qs: List[Q] = []
    rngq = random.Random(seed + 1)
    booked = [p for p in people if cal[p].slots] or people  # only ask about people the text mentions
    p = rngq.choice(booked)
    edges = {x for s_, e_, _ in cal[p].slots for x in (s_, e_)}
    h = rngq.choice([x for x in range(8, 18) if x not in edges] or list(range(8, 18)))  # never an endpoint
    busy = any(s <= h < e for s, e, _ in cal[p].slots)
    qs.append(Q("free_at", {"who": p, "hour": h}, ("bool", not busy), ("bool", busy), "flip"))
    if cal[p].slots:
        first = min(cal[p].slots)
        others = [t2 for _s, _e, t2 in cal[p].slots if t2 != first[2]]
        qs.append(Q("first_meeting", {"who": p}, ("title", first[2]),
                    ("title", rngq.choice(others) if others else rngq.choice([t for t in TITLES if t != first[2]])),
                    "wrong_entity"))
        qs.append(Q("count_meetings", {"who": p}, ("num", len(cal[p].slots)),
                    ("num", len(cal[p].slots) + rngq.choice([1, -1])), "off_by_one"))
    # Retrodiction, free in this domain: the `moved` event already records where the booking came from,
    # and nothing ever asked. The distractor is the CURRENT time, so answering needs the earlier state
    # rather than the latest mention -- which is the whole point of asking backwards.
    moves = [e for e in events if e.kind == "moved"]
    if moves:
        m = moves[len(moves) // 2].data
        qs.append(Q("slot_before_move", {"who": m["who"], "title": m["title"]},
                    ("hour", m["from_h"]), ("hour", m["to_h"]), "current"))
    a, b = rngq.sample(booked, 2) if len(booked) >= 2 else rngq.sample(people, 2)
    overlap = any(sa < eb and sb < ea for sa, ea, _ in cal[a].slots for sb, eb, _ in cal[b].slots)
    qs.append(Q("overlap", {"a": a, "b": b}, ("bool", overlap), ("bool", not overlap), "flip"))
    return Trace("schedule", seed, diff, w, events, qs)


DOMAINS = {"household": _household, "inventory": _inventory, "kinship": _kinship, "schedule": _schedule}


def sample_difficulty(rng: random.Random, p_fail: int = 0) -> Dict[str, int]:
    """`p_fail` is a PERCENTAGE, kept an int so it survives the difficulty dict's JSON round-trip in the
    generators' `meta`. 0 reproduces every trace generated before 2026-08-26 exactly."""
    return {
        "people": rng.randint(2, 5),
        "rooms": rng.randint(3, 5),
        "objects": rng.randint(3, 6),
        "ticks": rng.randint(4, 18),
        "p_fail": p_fail,
    }


def make_trace(domain: str, seed: int, difficulty: Optional[Dict[str, int]] = None) -> Trace:
    diff = difficulty or sample_difficulty(random.Random(seed ^ 0x5EED))
    return DOMAINS[domain](seed, diff)


def make_counterfactual(domain: str, seed: int, difficulty: Optional[Dict[str, int]] = None):
    """(factual, counterfactual) — the same world with a different final action.

    Ordinary narratives never isolate *why* an answer is what it is: the reader sees one history and one
    outcome. A pair that shares every byte but the last event, and disagrees on the answer, says which
    event was load-bearing. Returns None when the divert produced no change (nothing else was available
    from that state, or the alternative happened to give the same answer)."""
    diff = dict(difficulty or sample_difficulty(random.Random(seed ^ 0x5EED)))
    a = DOMAINS[domain](seed, diff)
    b = DOMAINS[domain](seed, {**diff, "divert": True})
    if len(a.events) != len(b.events) or a.events[:-1] != b.events[:-1] or a.events[-1] == b.events[-1]:
        a.world.close()
        b.world.close()
        return None
    return a, b
