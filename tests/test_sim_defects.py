"""Properties the sim audit of 2026-08-29 found broken: a tie is never asked as "who has more", the DPO rejected
answer is never literally true, a retrodiction question names one event, an object cannot be put into a
container in another room (and the refusal renders in every locale), a move names a room the world has, the
Russian refusal's pronouns agree with the object, and a zero p_fail draws nothing."""

import random

import pytest

from mote.sim.domains import DOMAINS, ROOMS, make_trace, sample_difficulty
from mote.sim.render import LOCALES, RU_OBJ_GENDER, _fail_ru
from mote.sim.tasks import parse_action


def _trace(domain, seed, p_fail=30):
    return make_trace(domain, seed, sample_difficulty(random.Random(seed ^ 0x5EED), p_fail=p_fail))


def _each(domain, seeds, p_fail=30):
    for s in seeds:
        t = _trace(domain, s, p_fail)
        try:
            yield t
        finally:
            t.world.close()


def test_who_has_more_is_never_a_tie():
    asked = 0
    for t in _each("inventory", range(1, 301)):
        for q in t.questions:
            if q.qtype == "who_has_more":
                asked += 1
                a, b, g = q.args["a"], q.args["b"], q.args["goods"]
                stock = {e.data["who"]: e.data for e in t.events if e.kind == "init_inventory"}  # may be absent
                assert q.answer[0] == "person" and q.wrong[0] == "person" and q.answer[1] != q.wrong[1]
    assert asked > 200  # the question is still asked for nearly every seed


def test_rejected_where_obj_is_never_the_containers_room():
    seen = 0
    for t in _each("household", range(1, 301)):
        cont_room = next(e.data["containers"] for e in t.events if e.kind == "init_household")
        for q in t.questions:
            if q.qtype == "where_obj" and q.answer[0] == "cont":
                seen += 1
                assert q.wrong[0] == "room" and q.wrong[1] != cont_room[q.answer[1][0]], (t.seed, q)
    assert seen > 30  # 41 container answers over 300 seeds


def test_retrodiction_anchors_are_unique():
    for t in _each("schedule", range(1, 201)):
        moves = [e for e in t.events if e.kind == "moved"]
        for q in t.questions:
            if q.qtype == "slot_before_move":
                same = [m for m in moves if (m.data["who"], m.data["title"]) == (q.args["who"], q.args["title"])]
                assert len(same) == 1, (t.seed, q)
                assert q.answer == ("hour", same[0].data["from_h"])
    for t in _each("household", range(1, 201)):
        for q in t.questions:
            if q.qtype == "where_obj_before":
                hits = [e for e in t.events if e.kind == q.args["verb"] and e.data.get("who") == q.args["who"] and e.data.get("obj") == q.args["obj"]]
                assert len(hits) == 1, (t.seed, q)


def test_remote_put_in_is_refused_and_rendered_everywhere():
    from mote.sim.domains import household_system
    from mote.sim.ecs import InRoom
    from mote.sim.render import LOCALES as R

    refused = 0
    for t in _each("household", range(1, 121)):
        w = t.world
        init = next(e.data for e in t.events if e.kind == "init_household")
        people, conts = list(init["people"]), init["containers"]
        for p in people:
            room = w.get(w.eid(p), InRoom).room
            far = [c for c, r in conts.items() if r != room]
            held = [o for o in init["objects"] if w.one(w.eid(o), "held_by") == w.eid(p)]
            if far and held:
                evs = household_system(w, [{"kind": "put_in", "who": p, "obj": held[0], "cont": far[0]}])
                assert evs and evs[-1].kind == "failed" and evs[-1].data["why"] == "cont_not_here", (t.seed, evs)
                for loc in ("en", "ru", "ja"):
                    s = R[loc]["event"](evs[-1])
                    assert s and far[0] in s.lower() or loc != "en", (loc, s)
                refused += 1
                break
    assert refused > 10


def test_move_to_a_room_the_world_lacks_is_unknown():
    checked = 0
    for t in _each("household", range(1, 61)):
        init = next(e.data for e in t.events if e.kind == "init_household")
        missing = [r for r in ROOMS if r not in init["rooms"]]
        who = next(iter(init["people"]))
        if missing:
            assert parse_action("household", f"{who}: move to {missing[0]}", t.world, init) is None
            assert parse_action("household", f"{who}: move to {init['rooms'][0]}", t.world, init) is not None
            checked += 1
    assert checked > 5


@pytest.mark.parametrize("obj,pron,was", [("key", "его", "он был"), ("apple", "его", "оно было"), ("book", "её", "она была")])
def test_russian_refusal_pronouns_agree_with_the_object(obj, pron, was):
    s = _fail_ru({"why": "not_here", "who": "Ivy", "obj": obj, "room": "cellar"})
    assert was in s, s
    s = _fail_ru({"why": "held_by_other", "who": "Ivy", "obj": obj, "holder": "Kofi"})
    assert s.endswith(pron + "."), s


def test_zero_p_fail_draws_no_failure():
    for dom in ("household", "inventory", "schedule"):
        for t in _each(dom, range(1, 41), p_fail=0):
            assert not any(e.kind == "failed" for e in t.events), (dom, t.seed)
