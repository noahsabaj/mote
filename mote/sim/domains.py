"""The four micro-domains. Each builds a world, scripts random actions, and derives questions
whose answers are read off the true component state (correct by construction). Every question
carries a plausible-wrong answer for DPO: a stale historical value or a same-type wrong entity —
the corruptions that test state tracking rather than surface form.

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


@dataclass
class Q:
    qtype: str
    args: Dict[str, Any]     # entity keys the renderer needs
    answer: Any              # structured truth (name key, number, bool)
    wrong: Any               # plausible-wrong for DPO
    wrong_kind: str          # "stale" | "wrong_entity" | "off_by_one"


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
class Held:
    by: str  # person name, "" = not held


@dataclass
class InContainer:
    container: str  # "" = loose


def _household_system(w: World, actions: List[Dict[str, Any]]) -> List[Event]:
    """The one dynamics system: applies move/take/put_in/put_down actions to components.
    The generator scripts the actions below; the agency layer may supply them instead."""
    events: List[Event] = []
    byname = {v: k for k, v in w.names.items()}

    def state(o):
        eid = byname[o]
        return w.get(eid, InRoom), w.get(eid, Held), w.get(eid, InContainer)

    for a in actions:
        who = a["who"]
        if a["kind"] == "move":
            w.get(byname[who], InRoom).room = a["to"]
            for o, (room, held, _c) in ((n, state(n)) for n in byname if w.get(byname[n], Held)):
                if held.by == who:
                    room.room = a["to"]
            events.append(Event(w.t, "move", {"who": who, "to": a["to"]}))
        elif a["kind"] == "take":
            room, held, cont = state(a["obj"])
            if held.by == "" and cont.container == "" and room.room == w.get(byname[who], InRoom).room:
                held.by = who
                events.append(Event(w.t, "take", {"who": who, "obj": a["obj"]}))
        elif a["kind"] == "put_in":
            _room, held, cont = state(a["obj"])
            if held.by == who:
                held.by = ""
                cont.container = a["cont"]
                events.append(Event(w.t, "put_in", {"who": who, "obj": a["obj"], "cont": a["cont"]}))
        elif a["kind"] == "put_down":
            room, held, _cont = state(a["obj"])
            if held.by == who:
                held.by = ""
                events.append(Event(w.t, "put_down", {"who": who, "obj": a["obj"], "room": room.room}))
    return events


def _household(seed: int, diff: Dict[str, int]) -> Trace:
    rng = random.Random(seed)
    w = World(seed)
    w.systems.append(_household_system)
    people = rng.sample(PEOPLE, diff["people"])
    rooms = rng.sample(ROOMS, diff["rooms"])
    objects = rng.sample(OBJECTS, diff["objects"])
    containers = rng.sample(CONTAINERS, min(2, diff["objects"]))
    byname: Dict[str, int] = {}
    for p in people:
        byname[p] = w.spawn(p)
        w.add(byname[p], InRoom(rng.choice(rooms)))
    cont_room = {c: rng.choice(rooms) for c in containers}
    for c in containers:
        byname[c] = w.spawn(c)
        w.add(byname[c], InRoom(cont_room[c]))
    for o in objects:
        byname[o] = w.spawn(o)
        w.add(byname[o], InRoom(rng.choice(rooms)))
        w.add(byname[o], Held(""))
        w.add(byname[o], InContainer(""))

    def snap(o):
        return {"room": w.get(byname[o], InRoom).room, "held": w.get(byname[o], Held).by,
                "cont": w.get(byname[o], InContainer).container}

    history: List[Tuple[int, str, Dict[str, str]]] = [(0, o, snap(o)) for o in objects]
    events: List[Event] = []
    for _ in range(diff["ticks"]):
        # script one action per tick from the true state, then let the SYSTEM apply it
        p = rng.choice(people)
        proom = w.get(byname[p], InRoom).room
        held_by_p = [o for o in objects if w.get(byname[o], Held).by == p]
        here = [o for o in objects if snap(o) == {"room": proom, "held": "", "cont": ""}]
        act = rng.random()
        if act < 0.4 or (not here and not held_by_p):
            action = {"kind": "move", "who": p, "to": rng.choice([r for r in rooms if r != proom])}
        elif (act < 0.7 and here) or not held_by_p:
            action = {"kind": "take", "who": p, "obj": rng.choice(here)}
        else:
            o = rng.choice(held_by_p)
            conts_here = [c for c in containers if cont_room[c] == proom]
            if conts_here and rng.random() < 0.5:
                action = {"kind": "put_in", "who": p, "obj": o, "cont": rng.choice(conts_here)}
            else:
                action = {"kind": "put_down", "who": p, "obj": o}
        new_events = w.step([action])
        events.extend(new_events)
        for e in new_events:
            if "obj" in e.data:
                history.append((e.t, e.data["obj"], snap(e.data["obj"])))
            elif e.kind == "move":
                for o in objects:
                    if w.get(byname[o], Held).by == e.data["who"]:
                        history.append((e.t, o, snap(o)))
    obj_state = {o: snap(o) for o in objects}
    loc = {p: w.get(byname[p], InRoom).room for p in people}

    qs: List[Q] = []
    rngq = random.Random(seed + 1)
    for o in rngq.sample(objects, min(len(objects), 3)):
        s = obj_state[o]
        first = next(h for h in history if h[1] == o)[2]
        if s["held"]:
            ans, wrong = ("held", s["held"]), ("held", rngq.choice([p for p in people if p != s["held"]]))
            wk = "wrong_entity"
        elif s["cont"]:
            ans = ("cont", s["cont"])
            wrong, wk = ("room", first["room"]), "stale"
        else:
            ans = ("room", s["room"])
            stale = first["room"] if first["room"] != s["room"] else rngq.choice([r for r in rooms if r != s["room"]])
            wrong, wk = ("room", stale), ("stale" if first["room"] != s["room"] else "wrong_entity")
        qs.append(Q("where_obj", {"obj": o}, ans, wrong, wk))
        if first["room"] != s["room"] or s["held"] or s["cont"]:
            qs.append(Q("where_obj_start", {"obj": o}, ("room", first["room"]),
                        ("room", s["room"] if not s["held"] and not s["cont"] else first["room"]) if not s["held"] and not s["cont"] else ("held", s["held"]) if s["held"] else ("cont", s["cont"]),
                        "stale"))
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


def _inventory(seed: int, diff: Dict[str, int]) -> Trace:
    rng = random.Random(seed)
    w = World(seed)
    people = rng.sample(PEOPLE, diff["people"])
    goods = rng.sample(GOODS, min(diff["objects"], len(GOODS)))
    price = {g: rng.randint(1, 5) for g in goods}
    stock: Dict[str, Stock] = {}
    for p in people:
        eid = w.spawn(p)
        st = Stock({g: rng.randint(0, 6) for g in goods}, rng.randint(10, 30))
        stock[p] = st
        w.add(eid, st)
    events: List[Event] = []
    start = {p: (dict(s.goods), s.coins) for p, s in stock.items()}
    for t in range(diff["ticks"]):
        buyer, seller = rng.sample(people, 2)
        g = rng.choice(goods)
        n = rng.randint(1, 3)
        if stock[seller].goods[g] >= n and stock[buyer].coins >= n * price[g]:
            stock[seller].goods[g] -= n
            stock[buyer].goods[g] += n
            cost = n * price[g]
            stock[buyer].coins -= cost
            stock[seller].coins += cost
            events.append(Event(t, "trade", {"buyer": buyer, "seller": seller, "goods": g, "n": n, "cost": cost}))
        else:
            g2 = rng.choice(goods)
            n2 = rng.randint(1, 2)
            stock[buyer].goods[g2] += n2
            events.append(Event(t, "harvest", {"who": buyer, "goods": g2, "n": n2}))
    qs: List[Q] = []
    rngq = random.Random(seed + 1)
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


# ---------------------------------------------------------------- kinship
@dataclass
class Parent:
    children: List[str] = field(default_factory=list)


@dataclass
class Spouse:
    of: str = ""


def _kinship(seed: int, diff: Dict[str, int]) -> Trace:
    rng = random.Random(seed)
    w = World(seed)
    n = max(6, diff["people"] + 4)
    names = rng.sample(KIN_NAMES, n)
    # three generations: g0 couples -> g1 children (some married) -> g2 children
    g0 = names[:2]
    g1 = names[2 : 2 + max(2, n // 3)]
    g2 = names[2 + max(2, n // 3) :]
    parents: Dict[str, Tuple[str, str]] = {}
    spouses: Dict[str, str] = {g0[0]: g0[1], g0[1]: g0[0]}
    for c in g1:
        parents[c] = (g0[0], g0[1])
    pairs = []
    g1_shuffled = g1[:]
    rng.shuffle(g1_shuffled)
    for i in range(0, len(g1_shuffled) - 1, 2):
        a, b = g1_shuffled[i], g1_shuffled[i + 1]
        spouses[a], spouses[b] = b, a
        pairs.append((a, b))
    for j, c in enumerate(g2):
        if not pairs:
            break
        a, b = pairs[j % len(pairs)]
        parents[c] = (a, b)
    for name in names:
        eid = w.spawn(name)
        w.add(eid, Parent([c for c, ps in parents.items() if name in ps]))
        w.add(eid, Spouse(spouses.get(name, "")))
    events = [Event(0, "family", {"parents": parents, "spouses": spouses, "gens": [g0, g1, g2]})]
    qs: List[Q] = []
    rngq = random.Random(seed + 1)
    g2_with_parents = [c for c in g2 if c in parents]
    if g2_with_parents:
        c = rngq.choice(g2_with_parents)
        mother = parents[c][0]
        gp = parents.get(mother, (None, None))[0]
        if gp:
            qs.append(Q("grandparent", {"who": c, "via": mother}, ("person", gp),
                        ("person", rngq.choice([x for x in g1 if x != mother])), "wrong_entity"))
        sibs = [x for x in g2_with_parents if x != c and parents[x] == parents[c]]
        qs.append(Q("count_siblings", {"who": c}, ("num", len(sibs)),
                    ("num", len(sibs) + rngq.choice([1, -1]) if sibs else 1), "off_by_one"))
    p = rngq.choice(g1)
    kids = [c for c, ps in parents.items() if p in ps]
    qs.append(Q("count_children", {"who": p}, ("num", len(kids)),
                ("num", len(kids) + rngq.choice([1, 2]) if not kids else len(kids) + rngq.choice([1, -1])), "off_by_one"))
    if spouses.get(p):
        qs.append(Q("spouse_of", {"who": p}, ("person", spouses[p]),
                    ("person", rngq.choice([x for x in names if x not in (p, spouses[p])])), "wrong_entity"))
    return Trace("kinship", seed, diff, w, events, qs)


# ---------------------------------------------------------------- schedule
@dataclass
class Meetings:
    slots: List[Tuple[int, int, str]] = field(default_factory=list)  # (start_h, end_h, title)


def _schedule(seed: int, diff: Dict[str, int]) -> Trace:
    rng = random.Random(seed)
    w = World(seed)
    people = rng.sample(PEOPLE, diff["people"])
    titles = ["standup", "review", "lunch", "planning", "call", "workshop"]
    cal: Dict[str, Meetings] = {}
    for p in people:
        eid = w.spawn(p)
        m = Meetings()
        cal[p] = m
        w.add(eid, m)
    events: List[Event] = []
    for t in range(diff["ticks"]):
        p = rng.choice(people)
        if cal[p].slots and rng.random() < 0.3:  # move a meeting
            i = rng.randrange(len(cal[p].slots))
            s, e, title = cal[p].slots[i]
            ns = rng.randint(8, 16)
            cal[p].slots[i] = (ns, ns + (e - s), title)
            events.append(Event(t, "moved", {"who": p, "title": title, "from_h": s, "to_h": ns}))
        else:
            s = rng.randint(8, 16)
            title = rng.choice(titles)
            cal[p].slots.append((s, s + rng.choice([1, 2]), title))
            events.append(Event(t, "booked", {"who": p, "title": title, "start_h": s, "end_h": cal[p].slots[-1][1]}))
    qs: List[Q] = []
    rngq = random.Random(seed + 1)
    p = rngq.choice(people)
    h = rngq.randint(8, 17)
    busy = any(s <= h < e for s, e, _ in cal[p].slots)
    qs.append(Q("free_at", {"who": p, "hour": h}, ("bool", not busy), ("bool", busy), "wrong_entity"))
    if cal[p].slots:
        first = min(cal[p].slots)
        others = [t2 for _s, _e, t2 in cal[p].slots if t2 != first[2]]
        qs.append(Q("first_meeting", {"who": p}, ("title", first[2]),
                    ("title", rngq.choice(others) if others else rngq.choice([t for t in titles if t != first[2]])),
                    "wrong_entity"))
        qs.append(Q("count_meetings", {"who": p}, ("num", len(cal[p].slots)),
                    ("num", len(cal[p].slots) + rngq.choice([1, -1])), "off_by_one"))
    a, b = rngq.sample(people, 2)
    overlap = any(sa < eb and sb < ea for sa, ea, _ in cal[a].slots for sb, eb, _ in cal[b].slots)
    qs.append(Q("overlap", {"a": a, "b": b}, ("bool", overlap), ("bool", not overlap), "wrong_entity"))
    return Trace("schedule", seed, diff, w, events, qs)


DOMAINS = {"household": _household, "inventory": _inventory, "kinship": _kinship, "schedule": _schedule}


def sample_difficulty(rng: random.Random) -> Dict[str, int]:
    return {
        "people": rng.randint(2, 5),
        "rooms": rng.randint(3, 5),
        "objects": rng.randint(3, 6),
        "ticks": rng.randint(4, 18),
    }


def make_trace(domain: str, seed: int, difficulty: Optional[Dict[str, int]] = None) -> Trace:
    diff = difficulty or sample_difficulty(random.Random(seed ^ 0x5EED))
    return DOMAINS[domain](seed, diff)
