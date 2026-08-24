"""The RLVR-1 environment (mote/sim/tasks.py): tasks are deterministic and verifiable, the expert reaches
every goal it was derived from (after pruning), the action language round-trips, the tool renders
observations in the task's locale and enforces the budget, and expert traces render as tool turns."""

import pytest

from mote.sim.tasks import (HELDOUT_SEED_BASE, TASK_DOMAINS, TEXT, SimEnv, action_text, expert_messages, fresh_world, goal_text,
                            heldout_tasks, init_of, legal_actions, make_task, parse_action)
from mote.tokenizer import CALL_ID, RESULT_ID, ByteTokenizer, ChatMessage


@pytest.mark.parametrize("domain", TASK_DOMAINS)
def test_legal_actions_round_trip_through_text(domain):
    tr = fresh_world(domain, 11)
    try:
        init = init_of(tr)
        legal = legal_actions(domain, tr.world, init)
        assert legal
        cal = lambda who, i: tr.world.get(tr.world.eid(who), __import__("mote.sim.domains", fromlist=["Calendar"]).Calendar).slots[i][2]  # noqa: E731
        for a in legal[:40]:
            text = action_text(a, cal)
            assert parse_action(domain, text.upper(), tr.world, init) == a, text
        assert parse_action(domain, "nobody: move to kitchen", tr.world, init) is None
        assert parse_action(domain, "gibberish", tr.world, init) is None
    finally:
        tr.world.close()


@pytest.mark.parametrize("seed", range(2_000_001, 2_000_031))
def test_expert_reaches_the_goal_and_is_pruned(seed):
    domain = TASK_DOMAINS[seed % len(TASK_DOMAINS)]
    locale = ("en", "ru", "ja")[seed % 3]
    t1, t2 = make_task(domain, seed, locale), make_task(domain, seed, locale)
    assert t1 == t2  # deterministic
    assert t1.goal and 1 <= len(t1.expert) <= t1.k_explored and t1.budget == len(t1.expert) + 2
    msgs = expert_messages(t1)  # asserts the goal holds after the expert's actions
    parts = msgs[1]["parts"]
    assert [p["type"] for p in parts] == ["call", "result"] * len(t1.expert) + ["text"]
    assert all(p["text"].startswith("sim: ") for p in parts[0::2][:-1])
    assert all(p["text"] != TEXT[locale]["nothing"] for p in parts[1::2])  # every expert action does something
    assert goal_text(t1.goal, locale) in t1.prompt and TEXT[locale]["instr"] in t1.prompt
    ids, mask = ByteTokenizer().format_chat_with_loss_mask([ChatMessage(m["role"], m["content"], m.get("parts")) for m in msgs])
    assert ids.count(CALL_ID) == len(t1.expert) == ids.count(RESULT_ID) and 0 < sum(mask) < len(mask)


def test_env_observations_budget_and_score():
    task = make_task("household", 2_000_010, "en", k=2)
    env = SimEnv(task)
    try:
        assert env.score()[0] is False or task.expert == []
        assert env.act("nobody does anything") == "Unknown action."
        # a well-formed action the system ignores (taking an object from another room) is a no-op
        who = task.expert[0].split(":")[0]
        far = next(o for o in ("key", "book", "lamp", "coin", "apple", "letter", "cup", "knife") if o in env.world.names.values()
                   and env.world.get(env.world.eid(o), __import__("mote.sim.domains", fromlist=["InRoom"]).InRoom).room
                   != env.world.get(env.world.eid(who), __import__("mote.sim.domains", fromlist=["InRoom"]).InRoom).room)
        assert env.act(f"{who}: take {far}") == "Nothing happened."
        for t in task.expert:
            assert env.act(t) not in ("Nothing happened.", "Unknown action.")
        ok, frac = env.score()
        assert ok and frac == 1.0
        left = task.budget - env.steps
        for _ in range(left):
            env.act("nobody: x")
        assert env.act(task.expert[0]) == "No moves left." and env.steps == task.budget
    finally:
        env.close()


def test_heldout_tasks_and_locale_goals():
    tasks = heldout_tasks(6)
    assert [t.locale for t in tasks] == ["en", "ru", "ja"] * 2 and all(t.seed > HELDOUT_SEED_BASE for t in tasks)
    for t in tasks:
        g = goal_text(t.goal, t.locale)
        assert g.startswith(TEXT[t.locale]["goal"]) and len(g) > len(TEXT[t.locale]["goal"]) + 3
        assert env_score_after_expert(t)


def env_score_after_expert(task) -> bool:
    env = SimEnv(task)
    try:
        for a in task.expert:
            env.act(a)
        return env.score()[0]
    finally:
        env.close()
