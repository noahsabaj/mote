"""The verifiable-world generator (mote/sim on esper): determinism, the step(actions) contract,
relationships, answer-by-construction, and every qtype rendering in all three locales."""

import json

from mote.sim.domains import DOMAINS, InRoom, Person, Portable, household_system, make_trace
from mote.sim.ecs import World
from mote.sim.render import LOCALES, narrative, qa_pairs


def test_same_seed_same_world_bytes():
    for domain in DOMAINS:
        a = make_trace(domain, 7)
        b = make_trace(domain, 7)
        assert a.world.serialize() == b.world.serialize(), domain
        assert [e.kind for e in a.events] == [e.kind for e in b.events], domain
        a.world.close(); b.world.close()


def test_household_routes_through_step_actions_and_relations():
    w = World(0)
    w.add_system(household_system)
    p, o = w.spawn("mara"), w.spawn("key")
    w.add(p, Person()); w.add(p, InRoom("kitchen"))
    w.add(o, Portable()); w.add(o, InRoom("kitchen"))
    ev = w.step([{"kind": "take", "who": "mara", "obj": "key"}])
    assert ev and ev[0].kind == "take" and w.one(o, "held_by") == p
    assert w.reverse(p, "held_by") == {o}
    w.step([{"kind": "move", "who": "mara", "to": "garden"}])
    assert w.get(o, InRoom).room == "garden"  # held objects travel
    assert w.step([{"kind": "take", "who": "mara", "obj": "key"}]) == []  # already held: refused, no event
    w.close()


def test_relations_transitive():
    w = World(1)
    a, b, c = w.spawn("a"), w.spawn("b"), w.spawn("c")
    w.relate(a, "inside", b); w.relate(b, "inside", c)
    assert w.transitive(a, "inside") == {b, c}
    assert w.reverse(c, "inside") == {b}
    w.unrelate(a, "inside", b)
    assert w.related(a, "inside") == set()
    w.close()


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
            t.world.close()
    assert len(seen) >= 15  # all question families exercised across seeds


def test_serialize_is_json_versioned_with_relations():
    t = make_trace("kinship", 3)
    state = json.loads(t.world.serialize().decode("utf-8"))
    assert state["schema"] == 2 and "components" in state and "names" in state
    assert "parent_of" in state["relations"] and "spouse_of" in state["relations"]
    t.world.close()
