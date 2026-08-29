"""The serving memory work of 2026-08-27 (nine findings, measured on Mote-138M at 16384).

Each test here pins one property that a measurement bought. The numbers in the comments are what was
measured on the 4060 Ti at the flagship shape; the assertions are the invariants that keep them true.
"""

import json

import pytest
import torch

from mote.config import resolve_preset
from mote.model.arena import RelationArena
from mote.model.hnet import HNetForCausalLM
from mote.runinfo import DEFAULT_BPIC, chunks_for, measured_bpic
from mote.serve.prefix_cache import PAGE, PrefixStore

CUDA = torch.cuda.is_available()


def _model(preset="mote-1m", seed=0, **dc):
    cfg = resolve_preset(preset)
    for k, v in dc.items():
        setattr(cfg.dc, k, v)
    torch.manual_seed(seed)
    return cfg, HNetForCausalLM(cfg).eval()


# --- 1. windowed prefill -----------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_windowed_read_produces_the_same_chunks_as_one_shot(seed):
    """The signed bar for windowing (2026-08-27): the boundary sequence has to be identical, because
    that is what decides the arena's contents. Measured at Mote-138M/16384 with 4096-byte windows:
    7,893 chunks both ways, peak 865 -> 252 MiB, 1556 -> 1534 ms."""
    cfg, model = _model(seed=seed)
    L, W = 512, 128
    ids = torch.randint(0, 256, (1, L), generator=torch.Generator().manual_seed(seed + 7))
    with torch.no_grad():
        one = model.allocate_inference_state("cpu")
        full = model.prefill(ids, one)
        win = model.allocate_inference_state("cpu")
        masks = [model.prefill(ids[:, :W], win, last_logits_only=True).routing.boundary_mask[0]]
        for a in range(W, L, W):
            masks.append(model.forward_from_state(ids[:, a:a + W], win, last_logits_only=True)[1])
    assert torch.equal(torch.cat(masks), full.routing.boundary_mask[0])
    assert win.main.n == one.main.n


def test_the_prefill_window_is_a_config_field_that_travels_with_the_checkpoint():
    cfg = resolve_preset("mote-138m")
    assert cfg.prefill_window == 4096
    from mote.config import MoteConfig

    cfg.prefill_window = 2048
    assert MoteConfig.from_dict(cfg.to_dict()).prefill_window == 2048


def test_an_engine_reads_a_long_prompt_in_windows():
    """`Engine._read` is the only path a prompt takes; a window of 0 is the old one-shot behaviour."""
    from mote.serve.engine import Engine

    cfg, model = _model("mote-32m")
    cfg.prefill_window = 64
    e = Engine.from_model(model, cfg, device="cpu")
    ids = list(torch.randint(0, 256, (300,)).tolist())
    with torch.no_grad():
        st = model.allocate_inference_state("cpu", arena=model.new_arena("cpu"))
        lg, bm = e._read(ids, st, fresh=True)
    assert lg.shape[-1] == cfg.pad_vocab_to and bm.shape[0] == len(ids)


# --- 2. last-row-only logits -------------------------------------------------------------------
def test_last_logits_only_returns_the_row_the_engine_actually_reads():
    """Every serving caller of prefill uses out.logits[0, -1]; the head over all L positions built a
    [1, 16384, 272] tensor (17 MiB) to throw away. The last row has to survive intact."""
    cfg, model = _model()
    ids = torch.randint(0, 256, (1, 128))
    with torch.no_grad():
        a = model.allocate_inference_state("cpu")
        full = model.prefill(ids, a).logits
        b = model.allocate_inference_state("cpu")
        last = model.prefill(ids, b, last_logits_only=True).logits
    assert last.shape == (1, 1, cfg.pad_vocab_to)
    V = cfg.vocab_size
    assert torch.allclose(full[0, -1, :V], last[0, -1, :V], atol=1e-3, rtol=1e-3)


def test_the_padding_rows_stay_masked_when_the_mask_is_applied_in_place():
    cfg, model = _model()
    with torch.no_grad():
        lg = model.head_logits(torch.randn(1, 4, cfg.d_model_outer))
    assert torch.isinf(lg[..., cfg.vocab_size:]).all() and (lg[..., cfg.vocab_size:] < 0).all()
    assert torch.isfinite(lg[..., : cfg.vocab_size]).all()


# --- 3. the arena is sized, not guessed --------------------------------------------------------
def test_capacity_comes_from_a_measured_rate_not_from_max_seq_len():
    """max_seq_len // 4 assumed 4 bytes a chunk. Three trained runs measured 3.2-3.45, so a full
    16384 context needed more rows than the default held and `ensure` fired mid-conversation."""
    old_default = 16384 // 4
    assert RelationArena.capacity_for(16384, 3.32) > old_default
    assert RelationArena.capacity_for(16384, 6.5) < old_default
    assert RelationArena.capacity_for(16384, 3.32) % 256 == 0


def test_new_arena_uses_bpic_when_it_has_one():
    cfg, model = _model("mote-32m")
    guessed = model.new_arena("cpu")
    measured = model.new_arena("cpu", bpic=3.32)
    assert measured.capacity == RelationArena.capacity_for(cfg.max_seq_len, 3.32)
    assert measured.capacity != guessed.capacity


def test_growth_with_nothing_valid_keeps_no_second_copy():
    """The 1296 MiB spike was old + new held together. With nothing to preserve there is nothing to
    copy, so the old rows go back first."""
    a = RelationArena(1, 1, 8, 2, "cpu", torch.float32)
    a.invalidate()
    assert a.ensure(9) and a.capacity == 16 and a.owner is None and a.n_valid == 0


def test_growth_preserves_the_rows_a_branch_owns():
    a = RelationArena(1, 1, 4, 2, "cpu", torch.float32)
    a.buf[:, :, :, :3] = 7.0
    a.owner, a.n_valid = 42, 3
    assert a.ensure(5) and a.capacity == 8
    assert float(a.buf[0, 0, 0, 2, 0]) == 7.0 and a.owner == 42


def test_rows_past_the_fill_are_finite():
    """Not merely "scratch": the decode graph multiplies masked (zero) weights into rows past S, and
    0 x NaN is NaN. Allocating with torch.empty broke test_graph_decode on 2026-08-27."""
    a = RelationArena(2, 2, 64, 4, "cpu", torch.float32)
    assert torch.isfinite(a.buf).all()
    a.owner, a.n_valid = 1, 4
    a.ensure(65)
    assert torch.isfinite(a.buf).all()


# --- 4. run metrics are read, not assumed ------------------------------------------------------
def test_measured_bpic_reads_the_last_eval_of_a_run(tmp_path):
    (tmp_path / "log.jsonl").write_text(
        json.dumps({"step": 1, "eval": {"val_bpb": 2.0, "val_bpic": 4.10}}) + "\n"
        + json.dumps({"step": 2, "train_bpb": 1.5}) + "\n"
        + json.dumps({"step": 3, "eval": {"val_bpb": 1.1, "val_bpic": 3.32}}) + "\n")
    assert measured_bpic(tmp_path) == pytest.approx(3.32)
    assert measured_bpic(tmp_path / "last.pt") == pytest.approx(3.32)
    assert chunks_for(tmp_path, 16384) == int(16384 / 3.32) + 1


def test_measured_bpic_is_none_when_a_run_recorded_nothing(tmp_path):
    assert measured_bpic(tmp_path) is None
    assert measured_bpic(tmp_path, default=None) is None
    assert chunks_for(tmp_path, 1000) == int(1000 / DEFAULT_BPIC) + 1


def test_a_truncated_log_does_not_break_a_checkpoint_load(tmp_path):
    (tmp_path / "log.jsonl").write_text('{"step": 1, "eval": {"val_bpic": 3.0}}\n{"step": 2, "ev')
    assert measured_bpic(tmp_path) == pytest.approx(3.0)


# --- 5. the prefix store ------------------------------------------------------------------------
def test_a_short_branch_costs_what_it_holds_not_a_whole_page():
    """27 MiB per branch at the flagship, whatever it held, capped the store at ~38 conversations."""
    arena = RelationArena(1, 1, 4 * PAGE, 2, "cpu", torch.float32)
    store = PrefixStore(1 << 30)
    b = store.commit(None, 0, "card", [1, 2], [torch.zeros(4)], torch.zeros(4), 3, arena)
    assert b.pages[0].shape[3] == 3
    full = store.commit(None, 0, "reply", [9] * 4, [torch.zeros(4)], torch.zeros(4), PAGE + 5, arena)
    assert full.pages[0].shape[3] == PAGE and full.pages[1].shape[3] == 5


def test_a_tail_page_grows_without_losing_what_it_held():
    arena = RelationArena(1, 1, 4 * PAGE, 2, "cpu", torch.float32)
    store = PrefixStore(1 << 30)
    arena.buf[:, :, :, :3] = 1.0
    b = store.commit(None, 0, "reply", [1], [torch.zeros(4)], torch.zeros(4), 3, arena)
    arena.buf[:, :, :, 3:9] = 2.0
    store.commit(b, 3, "reply", [1, 2], [torch.zeros(4)], torch.zeros(4), 9, arena)
    assert b.pages[0].shape[3] == 9
    assert float(b.pages[0][0, 0, 0, 0, 0]) == 1.0 and float(b.pages[0][0, 0, 0, 8, 0]) == 2.0


def test_eviction_spends_itself_on_branches_that_actually_free_pages():
    """A fork holds its parent's full pages by reference, so evicting the parent frees only its
    anchors — and the old loop, seeing the budget still unmet, took the fork as well (measured
    2026-08-27: 108 MiB over two branches, both evicted, store emptied)."""
    arena = RelationArena(1, 1, 4 * PAGE, 2, "cpu", torch.float32)
    store = PrefixStore(1 << 30)
    parent = store.commit(None, 0, "reply", list(range(4)), [torch.zeros(4)], torch.zeros(4), 2 * PAGE, arena)
    fork = store._fork(parent, 2 * PAGE, arena)
    store.branches.insert(0, fork)
    fork.ids, fork.n_chunks = b"\x00" * 8, 2 * PAGE
    other = store.commit(None, 0, "reply", list(range(90, 94)), [torch.zeros(4)], torch.zeros(4), 2 * PAGE, arena)
    parent.last_used, other.last_used, fork.last_used = 0.0, 1.0, 2.0
    assert store._exclusive_pages(parent) == 0 and store._exclusive_pages(other) > 0

    store.budget = int(store.used_bytes() * 0.75)
    store._evict()
    assert other not in store.branches       # the one holding pages of its own went
    assert fork in store.branches            # the hot fork survived
    assert parent in store.branches          # and so did the branch whose pages it shares


def test_the_pinned_card_is_never_evicted():
    arena = RelationArena(1, 1, PAGE, 2, "cpu", torch.float32)
    store = PrefixStore(1 << 30)
    card = store.commit(None, 0, "card", [1], [torch.zeros(4)], torch.zeros(4), 4, arena)
    store.commit(None, 0, "reply", [5, 5], [torch.zeros(4)], torch.zeros(4), 8, arena)
    store.budget = 1
    store._evict()
    assert store.branches == [card]


# --- 6. the decode graph ------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA, reason="graph decode is CUDA-only")
def test_the_rings_are_sized_for_a_reply_and_clamp_anything_larger():
    """ring_size was max_seq_len, 32x what one reply can reach: 7.45 MiB of device rings and as much
    pinned host memory at 16384, for 513 usable slots."""
    import threading

    from mote.serve.graph import GraphDecoder

    cfg, model = _model("mote-1m")
    model = model.cuda()
    arena = model.new_arena("cuda")
    gd = GraphDecoder(model, arena, "cuda", set(), ring_size=32)
    assert gd.ring_bytes.shape[0] == 32
    state = model.allocate_inference_state("cuda", arena=arena)
    with torch.no_grad():
        model.prefill(torch.randint(0, 256, (1, 16), device="cuda"), state)
    seen = []
    reason, _, _ = gd.run(state, torch.zeros(gd.V, device="cuda"), 0.0, 1.0,
                          10_000, threading.Event(), lambda recs: seen.extend(recs))
    assert len(seen) <= 32 - gd.K - 1  # served short rather than writing past the rings
    gd.close()


def test_the_capture_cache_keeps_the_newest_widths_and_drops_the_rest():
    """Bookkeeping only — no real captures. 24.4 MiB for the first capture and 16.25 MiB for each
    after, and a 16384-byte context at the measured chunk rate walks ~12 widths, so ~200 MiB of
    graphs would sit resident for the life of the engine."""
    from collections import OrderedDict

    from mote.serve.graph import GraphDecoder

    class _Stub:
        def __init__(self):
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    gd = GraphDecoder.__new__(GraphDecoder)
    gd.graphs, gd.max_graphs = OrderedDict(), 3
    made = {}
    for w in range(1, 6):
        made[w] = gd.graphs[(w * 256, 0)] = _Stub()
        gd._evict_graphs()
    assert [k[0] for k in gd.graphs] == [768, 1024, 1280]
    assert made[1].reset_called and made[2].reset_called   # evicted captures are destroyed, not leaked
    assert not made[5].reset_called

    gd.graphs[(768, 0)]  # a hit moves it to the end, so the LRU order is use order not insert order
    gd.graphs.move_to_end((768, 0))
    gd.graphs[(1536, 0)] = _Stub()
    gd._evict_graphs()
    assert [k[0] for k in gd.graphs] == [1280, 768, 1536]


def test_a_zero_cache_size_keeps_every_capture():
    from collections import OrderedDict

    from mote.serve.graph import GraphDecoder

    gd = GraphDecoder.__new__(GraphDecoder)
    gd.graphs, gd.max_graphs = OrderedDict(), 0
    for w in range(6):
        gd.graphs[(w, 0)] = object()
        gd._evict_graphs()
    assert len(gd.graphs) == 6


@pytest.mark.skipif(not CUDA, reason="graph decode is CUDA-only")
def test_a_real_capture_is_evicted_once_the_cache_is_full():
    """Two captures against a cache of one. Deliberately the smallest test that exercises real CUDA
    graphs: a capture's private pool is not returned to the allocator when the graph is reset, so a
    test that captures many of them starves everything that runs after it in the same process."""
    from mote.serve.graph import BUCKET, GraphDecoder

    cfg, model = _model("mote-1m")
    model = model.cuda()
    arena = model.new_arena("cuda", capacity=2 * BUCKET)
    gd = GraphDecoder(model, arena, "cuda", set(), ring_size=32)
    gd.max_graphs = 1
    state = model.allocate_inference_state("cuda", arena=arena)
    gd.load(state, torch.zeros(gd.V, device="cuda"))
    gd._graph(BUCKET)
    gd._graph(2 * BUCKET)
    assert len(gd.graphs) == 1 and next(iter(gd.graphs))[0] == 2 * BUCKET
    gd.close()
    del gd, state, arena, model
    torch.cuda.empty_cache()
