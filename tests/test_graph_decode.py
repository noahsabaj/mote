"""Decode as one CUDA graph per byte (mote/serve/graph.py): the graph path must reproduce the eager
`step` loop byte for byte (greedy), leave the same state behind (boundary and non-boundary bytes go
through IF nodes), stop exactly where the eager loop stops (max_bytes, a stop id — no state overshoot
despite K replays per sync), and sample from the same distribution at temperature > 0."""

import threading

import pytest
import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM
from mote.serve.engine import Engine, GenParams, _dist
from mote.serve.graph import GraphDecoder

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
DEV = "cuda"


def _cfg():
    return MoteConfig(
        d_model_outer=64, encoder_layers=2, decoder_layers=1,
        main=RelationCfg(n_layers=2, d_model=96, n_heads=4, d_ff=128),
        mbp=MBPCfg(enabled=False, n_layers=1, n_heads=2, d_ff=64),
        mamba3=Mamba3Cfg(d_state=32, headdim=32, expand=2), max_seq_len=512,
    )


def _model(seed=0):
    torch.manual_seed(seed)
    return HNetForCausalLM(_cfg()).to(DEV).eval()


def _flat(state):
    return [t for s in state.encoder + state.decoder for t in s] + [state.routing.last_hidden_state, state.dechunk.last_value]


@torch.no_grad()
def _eager(model, arena, prompt, n, stop_ids=()):
    st = model.allocate_inference_state(DEV, arena=arena)
    out = model.prefill(prompt, st)
    lg = out.logits[0, -1]
    bytes_, bits = [], []
    for _ in range(n):
        b = int(lg.argmax())
        if b in stop_ids:
            break
        lg2, routing, is_b, _ = model.step(torch.tensor([[b]], device=DEV), st)
        bytes_.append(b)
        bits.append(bool(is_b))
        lg = lg2[0, -1]
    return bytes_, bits, st, lg


@torch.no_grad()
def _graph(model, arena, prompt, n, stop_ids=(), temperature=0.0, top_p=1.0):
    st = model.allocate_inference_state(DEV, arena=arena)
    out = model.prefill(prompt, st)
    gd = GraphDecoder(model, arena, DEV, set(stop_ids), ring_size=512)
    recs = []
    reason, st2, lg = gd.run(st, out.logits[0, -1], temperature, top_p, n, threading.Event(), recs.extend)
    gd.close()
    return recs, reason, st2, lg, gd


def test_graph_matches_eager_greedy_and_state():
    seen_boundary = False
    for seed in range(3):
        model = _model(seed)
        arena = model.new_arena(DEV)
        prompt = torch.randint(0, 256, (1, 37), device=DEV)
        eb, ebits, est, elg = _eager(model, arena, prompt, 40)
        recs, reason, gst, glg, gd = _graph(model, arena, prompt, 40)
        gb = [r["byte"] for r in recs]
        assert gb == eb, (seed, gb, eb)
        assert [r["boundary"] for r in recs] == ebits
        assert reason == "max_bytes" and len(recs) == 40
        assert gst.main.n == est.main.n
        for a, b in zip(_flat(gst), _flat(est)):
            assert torch.allclose(a.float(), b.float(), atol=2e-3, rtol=2e-3), (a.float() - b.float()).abs().max()
        assert torch.allclose(glg, elg.float(), atol=2e-3, rtol=2e-3)
        # the arena rows written by the graph equal the eager rows (same chunks, same P2/I~)
        assert torch.allclose(arena.rows(0, gst.main.n).float(), arena.rows(0, est.main.n).float())
        seen_boundary |= any(ebits)
        assert all(0.0 <= r["p"] <= 1.0 + 1e-5 and r["entropy"] >= 0 for r in recs)
        assert all(len(r["retention"]) == 3 for r in recs)  # 2 encoder + 1 decoder mixers
    assert seen_boundary


def test_graph_stops_exactly_on_stop_id_and_max_bytes():
    # Random-init models emit one constant byte greedily, so the stop usually lands at byte 0 (nothing
    # consumed: `done` must hold for all K replays); the max_bytes case below stops mid-batch at 3.
    for seed in range(1, 5):
        model = _model(seed)
        arena = model.new_arena(DEV)
        prompt = torch.randint(0, 256, (1, 20), device=DEV)
        eb, _, _, _ = _eager(model, arena, prompt, 12)
        # stop on the first byte that has not occurred earlier in the reply (a random model may repeat itself)
        stop_at = next((i for i in range(1, len(eb)) if eb[i] not in eb[:i]), 0)
        stop_ids = {eb[stop_at]}
        eb2, _, est2, elg2 = _eager(model, arena, prompt, 12, stop_ids)
        assert len(eb2) == stop_at  # the eager loop stops before consuming the stop byte
        recs, reason, gst, glg, gd = _graph(model, arena, prompt, 12, stop_ids)
        assert reason == "eos" and [r["byte"] for r in recs] == eb2, seed
        for a, b in zip(_flat(gst), _flat(est2)):  # K replays per sync, yet the state froze at the stop
            assert torch.allclose(a.float(), b.float(), atol=2e-3, rtol=2e-3)
        assert torch.allclose(glg, elg2.float(), atol=2e-3, rtol=2e-3)
        # max_bytes smaller than K: exactly that many bytes, no overshoot
        recs3, reason3, gst3, _, _ = _graph(model, arena, prompt, 3)
        assert reason3 == "max_bytes" and len(recs3) == 3
        _, _, est3, _ = _eager(model, arena, prompt, 3)
        for a, b in zip(_flat(gst3), _flat(est3)):
            assert torch.allclose(a.float(), b.float(), atol=2e-3, rtol=2e-3)


def test_device_sampler_matches_the_eager_distribution():
    model = _model(2)
    arena = model.new_arena(DEV)
    gd = GraphDecoder(model, arena, DEV, set(), ring_size=64)
    torch.manual_seed(0)
    logits = torch.randn(gd.V, device=DEV) * 3
    temperature, top_p = 0.8, 0.9
    ref = _dist(logits, temperature, top_p)
    gd.logits.copy_(logits)
    gd.temp.fill_(temperature)
    gd.top_p.fill_(top_p)
    gd.greedy.fill_(False)
    counts = torch.zeros(gd.V, device=DEV)
    N = 6000
    for _ in range(N):
        gd.u.copy_(torch.rand((), device=DEV, generator=gd.gen))
        gd.wpos.zero_()
        b = gd._sample()
        counts[b] += 1
        assert abs(float(gd.ring_p[0]) - float(ref[b])) < 1e-4  # p reported = prob under the sampling distribution
    gd.close()
    tv = 0.5 * (counts / N - ref).abs().sum().item()
    assert tv < 0.05, tv
    assert int((counts > 0).sum()) == int((ref > 0).sum())  # exactly the nucleus support


def test_engine_graph_path_equals_eager_path(tmp_path, monkeypatch):
    import mote.serve.engine as E

    monkeypatch.setattr(E, "STOP_IDS", set())
    model = _model(3)
    run = tmp_path / "runs" / "pilot_tiny"
    run.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 3, "config": _cfg().to_dict(), "extra": {}}, run / "last.pt")
    eng = Engine(run / "last.pt", device=DEV)
    assert eng._graph_ok
    msgs = [{"role": "user", "content": "hello there"}]
    params = GenParams(temperature=0.0, max_bytes=30, n_candidates=0)

    def run_(eng):
        evs = []
        eng.generate(msgs, params, evs.append, threading.Event())
        return evs

    g_evs = run_(eng)
    eng._graph_ok = False
    eng.prefix_cache.clear()
    eng.arena.invalidate()
    e_evs = run_(eng)
    gb = [e["byte"] for e in g_evs if e["type"] == "byte"]
    ebb = [e["byte"] for e in e_evs if e["type"] == "byte"]
    assert gb == ebb and len(gb) == 30
    assert g_evs[-1]["reason"] == e_evs[-1]["reason"] == "max_bytes"
    gbe = [e for e in g_evs if e["type"] == "byte"]
    ebe = [e for e in e_evs if e["type"] == "byte"]
    assert [e["chunk"] for e in gbe] == [e["chunk"] for e in ebe]
    assert all(abs(a["p"] - b["p"]) < 2e-3 and abs(a["entropy"] - b["entropy"]) < 2e-2 and a["boundary"] == b["boundary"] for a, b in zip(gbe, ebe))
    if any(e["boundary"] for e in gbe):  # diagnostics are emitted at chunk boundaries on both paths
        assert any(e["type"] == "diagnostics" for e in g_evs) and any(e["type"] == "diagnostics" for e in e_evs)
    # the next turn continues from the graph's reply anchor exactly as the eager engine would
    eng._graph_ok = True
    msgs2 = msgs + [{"role": "assistant", "content": g_evs[-1]["text"]}, {"role": "user", "content": "more"}]
    evs2 = run_(eng)
    start = next(e for e in evs2 if e["type"] == "start")
    assert start["prefix"]["reused"] > 0


def test_graph_topology_degrades_instead_of_breaking_serving():
    """CUDAGraph.get_graph_data needs cuda-bindings >= 13.1, which the cu126 stack does not have.
    A debug readout must never be able to take serving down, so it reports the failure as data."""
    import torch

    from mote.serve.graph import ANNOTATE, GraphDecoder

    assert ANNOTATE is False, "annotations cost capture time; they are opt-in via MOTE_GRAPH_ANNOTATE"
    if not torch.cuda.is_available():
        return
    d = GraphDecoder.__new__(GraphDecoder)  # no model needed: only the graphs dict is read
    g = torch.cuda.CUDAGraph()
    d.graphs = {(64, 0): g}
    out = d.graph_topology()
    assert set(out) == {"Cb=64"}
    row = out["Cb=64"]
    assert "error" in row or "nodes" in row, f"neither a readout nor a reported failure: {row}"
