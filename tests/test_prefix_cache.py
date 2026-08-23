"""Prefix-state cache (mote/serve/prefix_cache.py, Engine._prefill_with_cache): a warm continuation
equals a cold prefill up to float rounding, the cache honours its budget and picks the longest usable
snapshot, and the engine reports what it reused and verifies it on request."""

import threading

import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM
from mote.serve.engine import Engine, GenParams
from mote.serve.prefix_cache import PrefixCache


def _cfg():
    return MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )


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
        parked = HNetForCausalLM.move_state(warm, "cpu")  # the cache's round trip is a copy, never an alias
        lg, bm, _ = model.forward_from_state(ids[:, 23:], parked)
        assert bm.tolist() == out.routing.boundary_mask[0, 23:].tolist(), seed
        assert torch.allclose(lg[0, -1], out.logits[0, -1], atol=1e-4, rtol=1e-4), (lg[0, -1] - out.logits[0, -1]).abs().max()
        nb = torch.tensor([[65]])
        a, _, _, _ = model.step(nb, cold)
        b, _, _, _ = model.step(nb, parked)
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)
        # the source of a moved state is untouched by continuing the copy: a second copy continues identically
        parked2 = HNetForCausalLM.move_state(warm, "cpu")
        lg2, bm2, _ = model.forward_from_state(ids[:, 23:], parked2)
        assert bm2.tolist() == bm.tolist() and torch.allclose(lg2, lg, atol=1e-5, rtol=1e-5)


def test_prefix_cache_picks_longest_prefix_and_keeps_budget():
    c = PrefixCache(budget_bytes=4096)
    st = lambda n: [torch.zeros(n, dtype=torch.uint8)]
    lg = torch.zeros(4)
    c.put("card", [1, 2, 3], st(100), lg, 1)
    c.put("prompt", [1, 2, 3, 4, 5], st(100), lg, 2)
    c.put("reply", [1, 2, 3, 9], st(100), lg, 2)
    assert c.peek([1, 2, 3, 4, 5, 6]).n_ids == 5
    assert c.peek([1, 2, 3, 9, 9]).n_ids == 4
    assert c.peek([1, 2, 3]).kind == "card"
    assert c.peek([1, 2]) is None and c.peek([7, 7, 7, 7]) is None
    assert c.lookup([1, 2, 3, 4, 5]).n_ids == 5 and c.hits == 1 and c.items[0].n_ids == 5
    c.put("prompt", [1, 2, 3, 4, 5], st(100), lg, 2)  # same bytes: replaced, not duplicated
    assert len(c.items) == 3
    c.put("reply", [5, 5, 5], st(3900), lg, 1)  # over budget: least recently used snapshots go
    assert c.used <= 4096 and c.peek([5, 5, 5, 1]) is not None and len(c.items) < 4
    assert c.put("big", [8], st(5000), lg, 0) is None  # larger than the whole budget: not stored
    assert PrefixCache(0).put("x", [1], st(1), lg, 0) is None
    assert c.report()["snapshots"] == len(c.items)


def test_engine_reuses_the_previous_turn_and_verifies(tmp_path, monkeypatch):
    import mote.serve.engine as E

    monkeypatch.setattr(E, "STOP_IDS", set())  # a random-init model samples EOS at once; keep the stream going
    eng = _engine(tmp_path)
    params = GenParams(temperature=0.0, max_bytes=10, n_candidates=0)

    def run(messages, context=None):
        evs = []
        eng.generate(messages, params, evs.append, threading.Event(), context=context)
        return evs, next(e for e in evs if e["type"] == "start"), next(e for e in evs if e["type"] == "done")

    msgs = [{"role": "system", "content": "You are a test."}, {"role": "user", "content": "hello there"}]
    _, start1, done1 = run(msgs)
    assert start1["prefix"]["reused"] == 0 and start1["prefix"]["prefilled"] == start1["prompt_bytes"]
    kinds = {s.kind for s in eng.prefix_cache.items}
    assert kinds == {"card", "prompt", "reply"}

    msgs2 = msgs + [{"role": "assistant", "content": done1["text"]}, {"role": "user", "content": "and again"}]
    evs2, start2, done2 = run(msgs2, context={"verify_prefix": True})
    assert start2["prefix"]["reused"] >= start1["prompt_bytes"]  # at least the whole first prompt came from the cache
    assert start2["prefix"]["prefilled"] == start2["prompt_bytes"] - start2["prefix"]["reused"]
    check = next(e["prefix_check"] for e in evs2 if e["type"] == "diagnostics" and "prefix_check" in e)
    assert check["boundary_flips"] == 0 and check["chunks_cold"] == check["chunks_warm"]
    assert check["max_logit_diff"] < 1e-3

    # a regenerate finds the prompt snapshot itself: nothing to read
    _, start3, done3 = run(msgs2)
    assert start3["prefix"]["reused"] == start3["prompt_bytes"] and start3["prefix"]["prefilled"] == 0
    assert done3["text"] == done2["text"]  # greedy, so the warm and the re-warm replies agree

    # a different conversation falls back to the shared card snapshot
    other = [msgs[0], {"role": "user", "content": "something else entirely"}]
    _, start4, _ = run(other)
    card_len = len(eng._card_ids(other))
    assert start4["prefix"]["reused"] == card_len

    # budget 0 disables the cache without changing the output
    eng.prefix_cache = PrefixCache(0)
    _, start5, done5 = run(msgs2)
    assert start5["prefix"]["reused"] == 0 and done5["text"] == done2["text"]
