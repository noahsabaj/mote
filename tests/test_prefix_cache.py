"""Prefix store (mote/infer/prefix_cache.py, Engine._read_prompt): a warm continuation equals a cold
prefill up to float rounding, anchors cost only the small states (the arena rows live in pages), branches
extend / refresh / fork correctly and honour the budget, the hot arena is not re-copied, and the engine
reports what it reused and verifies it on request."""

import threading

import torch

from mote.model.arena import RelationArena
from mote.model.hnet import HNetForCausalLM
from mote.infer.engine import Engine, GenParams
from mote.infer.prefix_cache import PAGE, PrefixStore, state_nbytes
from conftest import tiny_cfg


def _cfg():
    return tiny_cfg()


def _model(seed: int) -> HNetForCausalLM:
    torch.manual_seed(seed)
    return HNetForCausalLM(_cfg()).eval()


def _engine(tmp_path) -> Engine:
    model = _model(0)
    run = tmp_path / "runs" / "pilot_tiny"
    run.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 3, "config": _cfg().to_dict(), "extra": {"bytes_seen": 3000}}, run / "last.pt")
    return Engine(run / "last.pt", device="cpu")


@torch.no_grad()
def test_split_prefill_matches_cold_prefill():
    g = torch.Generator().manual_seed(5)
    for seed in range(4):
        model = _model(seed)
        ids = torch.randint(0, 256, (1, 40), generator=g)
        cold = model.allocate_inference_state("cpu")
        out = model.prefill(ids, cold)
        warm = model.allocate_inference_state("cpu")
        model.prefill(ids[:, :23], warm)
        parked = HNetForCausalLM.move_state(warm, "cpu")  # the anchor's round trip is a copy, never an alias
        lg, bm, _ = model.forward_from_state(ids[:, 23:], parked)
        assert bm.tolist() == out.routing.boundary_mask[0, 23:].tolist(), seed
        assert torch.allclose(lg[0, -1], out.logits[0, -1], atol=1e-4, rtol=1e-4), (lg[0, -1] - out.logits[0, -1]).abs().max()
        assert parked.main.n == cold.main.n
        nb = torch.tensor([[65]])
        a, _, _ = model.step(nb, cold)
        b, _, _ = model.step(nb, parked)
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)
        # the source of a moved state is untouched by continuing the copy: a second copy continues identically
        parked2 = HNetForCausalLM.move_state(warm, "cpu")
        lg2, bm2, _ = model.forward_from_state(ids[:, 23:], parked2)
        assert bm2.tolist() == bm.tolist() and torch.allclose(lg2, lg, atol=1e-5, rtol=1e-5)


def _fill(arena: RelationArena, n: int, value: float):
    arena.buf[:, :, :, :n] = value


def test_store_branches_anchors_pages_and_budget():
    arena = RelationArena(n_layers=1, n_heads=1, capacity=4 * PAGE, d_head=2, device="cpu", dtype=torch.float32)
    st = lambda: [torch.zeros(64)]  # a stand-in for the CPU state (256 B)
    lg = torch.zeros(4)
    store = PrefixStore(budget_bytes=1 << 30)
    # cold read of the card: 10 chunks, pinned branch
    _fill(arena, 10, 1.0)
    card = store.commit(None, 0, "card", [1, 2, 3], st(), lg, 10, arena)
    assert card.pinned and card.n_chunks == 10 and len(card.pages) == 1 and arena.owner == card.id
    # a conversation continues the card: forks from it (the card branch is never extended)
    _fill(arena, 300, 2.0)
    conv = store.commit(card, 10, "prompt", [1, 2, 3, 4, 5], st(), lg, 300, arena)
    assert conv is not card and conv.n_chunks == 300 and len(conv.pages) == 2 and card.n_chunks == 10
    assert store.peek([1, 2, 3, 4, 5, 6]).n_ids == 5 and store.peek([1, 2, 3]).anchor.kind == "card"
    assert store.peek([1, 2]) is None and store.peek([7, 7, 7, 7]) is None
    # the reply extends the conversation in place: rows appended, same branch
    _fill(arena, 400, 3.0)
    reply = store.commit(conv, 300, "reply", [1, 2, 3, 4, 5, 9], st(), lg, 400, arena)
    assert reply is conv and conv.n_chunks == 400 and [a.kind for a in conv.anchors] == ["prompt", "reply"]
    # a regenerate: the prompt anchor is an exact hit, refreshed in place, nothing copied
    hit = store.lookup([1, 2, 3, 4, 5])
    assert hit.branch is conv and hit.anchor.kind == "prompt"
    same = store.commit(conv, 300, "prompt", [1, 2, 3, 4, 5], st(), lg, 300, arena)
    assert same is conv and len(conv.anchors) == 2
    # ... and a different reply forks at the prompt's rows: shared full page, copied partial page
    _fill(arena, 350, 4.0)
    fork = store.commit(conv, 300, "reply", [1, 2, 3, 4, 5, 8], st(), lg, 350, arena)
    assert fork is not conv and fork.n_chunks == 350 and fork.pages[0] is conv.pages[0] and fork.pages[1] is not conv.pages[1]
    assert torch.equal(fork.pages[1][..., : 300 - PAGE, :], conv.pages[1][..., : 300 - PAGE, :])
    assert float(fork.pages[1][0, 0, 0, 300 - PAGE, 0]) == 4.0 and float(conv.pages[1][0, 0, 0, 300 - PAGE, 0]) == 3.0
    # hydrate: the arena is hot for the fork (it just committed) -> no rows copied; another branch -> copied
    before = store.rows_copied_in
    store.hydrate(fork, 350, arena)
    assert store.rows_copied_in == before and arena.owner == fork.id
    arena.buf.zero_()
    arena.invalidate()
    store.hydrate(conv, 400, arena)
    assert store.rows_copied_in == before + 400 and float(arena.buf[0, 0, 0, 399, 0]) == 3.0 and float(arena.buf[0, 0, 0, 5, 0]) == 1.0
    # budget: shared pages are counted once; eviction drops whole unpinned branches, never the card
    page_bytes = conv.pages[0].numel() * conv.pages[0].element_size()
    assert store.used_bytes() < 5 * page_bytes + 6 * state_nbytes(st()) + 6 * 16
    # A full page is PAGE rows — that is what makes it shareable — but a branch's TAIL page holds
    # only the rows it has, so the card's 10 chunks cost 10 rows and not 256 (2026-08-27: allocating
    # every page full meant a one-chunk branch cost 27 MiB at the flagship). Budget the eviction at
    # exactly what the card weighs, so nothing but the card can fit.
    assert card.pages[0].shape[3] == 10 and conv.pages[0].shape[3] == PAGE and conv.pages[1].shape[3] == 400 - PAGE
    card_bytes = sum(p.numel() * p.element_size() for p in card.pages) + sum(a.nbytes for a in card.anchors)
    store.budget = card_bytes
    store._evict()
    assert card in store.branches and all(b.pinned for b in store.branches)
    assert PrefixStore(0).commit(None, 0, "card", [1], st(), lg, 1, arena) is None
    assert store.report()["snapshots"] == sum(len(b.anchors) for b in store.branches)


def test_engine_reuses_the_previous_turn_and_verifies(tmp_path, monkeypatch):
    import mote.infer.engine as E

    monkeypatch.setattr(E, "STOP_IDS", set())  # a random-init model samples EOS at once; keep the stream going
    eng = _engine(tmp_path)
    params = GenParams(temperature=0.0, max_bytes=10)

    def run(messages, context=None):
        evs = []
        eng.generate(messages, params, evs.append, threading.Event(), context=context)
        return evs, next(e for e in evs if e["type"] == "start"), next(e for e in evs if e["type"] == "done")

    msgs = [{"role": "system", "content": "You are a test."}, {"role": "user", "content": "hello there"}]
    _, start1, done1 = run(msgs)
    assert start1["prefix"]["reused"] == 0 and start1["prefix"]["prefilled"] == start1["prompt_bytes"]
    kinds = {a.kind for b in eng.prefix_cache.branches for a in b.anchors}
    assert kinds == {"card", "prompt", "reply"}
    card_branch = next(b for b in eng.prefix_cache.branches if b.pinned)
    assert len(card_branch.anchors) == 1 and card_branch.anchors[0].kind == "card"
    # an anchor holds the small states only: far smaller than the arena rows it would have copied before
    anchor = max((a for b in eng.prefix_cache.branches for a in b.anchors), key=lambda a: a.n_ids)
    assert anchor.nbytes < 0.05 * eng.arena.nbytes() + 64 * 1024

    msgs2 = msgs + [{"role": "assistant", "content": done1["text"]}, {"role": "user", "content": "and again"}]
    copied = eng.prefix_cache.rows_copied_in
    evs2, start2, done2 = run(msgs2, context={"verify_prefix": True})
    assert start2["prefix"]["reused"] >= start1["prompt_bytes"]  # at least the whole first prompt came from the store
    assert start2["prefix"]["prefilled"] == start2["prompt_bytes"] - start2["prefix"]["reused"]
    assert eng.prefix_cache.rows_copied_in == copied  # the arena was hot: nothing hydrated
    check = next(e["prefix_check"] for e in evs2 if e["type"] == "diagnostics" and "prefix_check" in e)
    assert check["boundary_flips"] == 0 and check["chunks_cold"] == check["chunks_warm"]
    assert check["max_logit_diff"] < 1e-3

    # a regenerate finds the prompt anchor itself: nothing to read
    _, start3, done3 = run(msgs2)
    assert start3["prefix"]["reused"] == start3["prompt_bytes"] and start3["prefix"]["prefilled"] == 0
    assert done3["text"] == done2["text"]  # greedy, so the warm and the re-warm replies agree

    # a different conversation falls back to the shared card anchor, and hydrating it copies rows
    other = [msgs[0], {"role": "user", "content": "something else entirely"}]
    _, start4, _ = run(other)
    card_len = len(eng._card_ids(other))
    assert start4["prefix"]["reused"] == card_len

    # a swap re-warms the recent conversations: their anchors come back without any user message
    eng.prefix_cache.clear()
    eng.arena.invalidate()
    assert eng.prefix_cache.peek(eng.tok.format_chat([E.ChatMessage(m["role"], m["content"]) for m in msgs2], add_generation_prompt=True)) is None
    # (rewarm plans from the store's own records, so re-run one turn first to have a branch to plan from)
    run(msgs2)
    plan = eng.prefix_cache.rewarm_plan(600.0, 3)
    assert plan and plan[0][1][-1][0] == "reply"
    rep = eng.rewarm()
    assert rep["branches"] >= 1
    _, start5, done5 = run(msgs2)
    assert start5["prefix"]["reused"] == start5["prompt_bytes"] and done5["text"] == done2["text"]

    # budget 0 disables the store without changing the output
    eng.prefix_cache = PrefixStore(0)
    _, start6, done6 = run(msgs2)
    assert start6["prefix"]["reused"] == 0 and done6["text"] == done2["text"]
