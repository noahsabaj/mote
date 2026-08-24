"""The verifiable-world generator (mote/sim): determinism, the step(actions) contract,
answer-by-construction, and every qtype rendering in all three locales."""

import json

from mote.sim.domains import DOMAINS, make_trace
from mote.sim.ecs import World
from mote.sim.render import LOCALES, narrative, qa_pairs


def test_same_seed_same_world_bytes():
    for domain in DOMAINS:
        a = make_trace(domain, 7)
        b = make_trace(domain, 7)
        assert a.world.serialize() == b.world.serialize(), domain
        assert [e.kind for e in a.events] == [e.kind for e in b.events], domain


def test_household_routes_through_step_actions():
    w = World(0)
    from mote.sim.domains import Held, InContainer, InRoom, _household_system

    w.systems.append(_household_system)
    p, o = w.spawn("mara"), w.spawn("key")
    w.add(p, InRoom("kitchen"))
    w.add(o, InRoom("kitchen"))
    w.add(o, Held(""))
    w.add(o, InContainer(""))
    ev = w.step([{"kind": "take", "who": "mara", "obj": "key"}])
    assert ev and ev[0].kind == "take" and w.get(o, Held).by == "mara"
    ev = w.step([{"kind": "move", "who": "mara", "to": "garden"}])
    assert w.get(o, InRoom).room == "garden"  # held objects travel
    ev = w.step([{"kind": "take", "who": "mara", "obj": "key"}])
    assert ev == []  # already held: the system refuses, no event


def test_answers_disagree_with_wrongs_and_render_everywhere():
    seen = set()
    for domain in DOMAINS:
        for seed in range(30):
            t = make_trace(domain, seed)
            for q in t.questions:
                assert q.answer != q.wrong, (domain, seed, q.qtype)
                seen.add(q.qtype)
            for loc in LOCALES:
                doc = narrative(t, loc)
                assert doc.strip(), (domain, seed, loc)
                for p in qa_pairs(t, loc):
                    assert p["question"].strip() and p["answer"].strip() and p["wrong"].strip()
                    assert p["answer"] != p["wrong"]
    assert len(seen) >= 14  # all question families exercised across seeds


def test_serialize_is_json_and_versioned():
    t = make_trace("household", 3)
    state = json.loads(t.world.serialize().decode("utf-8"))
    assert state["schema"] == 1 and "components" in state and "names" in state
