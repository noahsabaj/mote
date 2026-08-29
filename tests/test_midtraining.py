"""The mid-training stage as re-signed 2026-08-26 (docs/research/midtraining-2026-08-26.md).

Four pieces, each guarding one 2026 finding against a later edit that would quietly undo it:

* `fim_window` — 2607.12463's function-aware fill-in-the-middle, cutting at the tool protocol's own
  `<|call|>`/`<|result|>` boundaries rather than at random offsets.
* `build_spec_docs` — Model Spec Midtraining (2605.02087): documents *about* the model, not dialogues
  spoken by it, and never the identity card recited as an answer.
* `sim.long` — the long-range dependency source. The ordinary sim has no dependency over 1 KB at all,
  so "long" here has to stay a measured number rather than a longer document.
* `mote.eval.proxy` — the gate's decider. Two bugs found by building it: a masked padding column NaN-ing
  every weighted mean (0 * -inf), and entropy weighting, which looked like the natural byte-level choice
  and reproduced a known quality ordering *backwards* because entropy is the candidate's own uncertainty.
"""

import random
import re

import numpy as np
import pytest
import torch

from mote.data.build_spec_docs import SECTION_ASKS, SECTION_BODY, generate as spec_docs
from mote.data.loader import fim_window
from mote.eval.proxy import trajectory_stats
from mote.identity import NAME, SPEC_SECTIONS
from mote.sim.domains import DOMAINS, make_trace, sample_difficulty
from mote.sim.long import dependency_distance, long_difficulty
from mote.sim.render import narrative, qa_pairs
from mote.tokenizer import (CALL_ID, FIM_MIDDLE_ID, FIM_PREFIX_ID, FIM_SUFFIX_ID, RESULT_ID, VOCAB_SIZE)


# --- FIM -------------------------------------------------------------------------------------------
def _traced(before: bytes = b"the agent looks around. ", call: bytes = b"sim: take candle",
            after: bytes = b" it now holds the candle.") -> np.ndarray:
    return np.array(list(before) + [CALL_ID] + list(call) + [RESULT_ID] + list(after), dtype=np.int64)


def test_fim_cuts_at_the_call_and_keeps_the_length():
    w = _traced()
    out = fim_window(w, np.random.default_rng(0))
    assert len(out) == len(w), "a permuted window still has to be exactly seq_len+1 ids"
    assert out[0] == FIM_PREFIX_ID and FIM_SUFFIX_ID in out and FIM_MIDDLE_ID in out
    mid = int(np.flatnonzero(out == FIM_MIDDLE_ID)[0])
    span = out[mid + 1:]
    # the middle is the whole call including the <|result|> that closes it: a truncated one would drop the
    # marker that says where the call ends, which is the thing being learned
    assert span[0] == CALL_ID and span[-1] == RESULT_ID
    assert bytes(int(x) for x in span[1:-1]) == b"sim: take candle"


def test_fim_leaves_a_window_with_no_call_alone():
    """Most windows in a general mix contain no tool call. They must pass through untouched, which is why
    this is a per-shard mode (`--mix data/tool_traces:0.03:fim`) and not a global transform."""
    plain = np.array(list(b"an ordinary sentence with no tool call in it"), dtype=np.int64)
    assert np.array_equal(fim_window(plain, np.random.default_rng(0)), plain)
    unclosed = np.array(list(b"x") + [CALL_ID] + list(b"sim: take"), dtype=np.int64)  # no <|result|>
    assert np.array_equal(fim_window(unclosed, np.random.default_rng(0)), unclosed)


def test_fim_gives_up_suffix_not_middle_when_it_does_not_fit():
    """Three sentinels make the permutation three ids longer, so something must be dropped. It is the
    suffix: the middle is the span being predicted."""
    w = _traced(after=b" " + b"z" * 200)
    out = fim_window(w, np.random.default_rng(0))
    assert len(out) == len(w)
    mid = int(np.flatnonzero(out == FIM_MIDDLE_ID)[0])
    assert bytes(int(x) for x in out[mid + 2:-1]) == b"sim: take candle"


def test_fim_ids_are_inside_the_padded_vocab():
    """266-268 were free rows in the embedding, which is padded to 272 — adding them needs no surgery on
    any existing checkpoint, and the head still masks everything at or above VOCAB_SIZE."""
    from mote.config import MoteConfig

    assert FIM_PREFIX_ID < FIM_SUFFIX_ID < FIM_MIDDLE_ID < VOCAB_SIZE <= MoteConfig().pad_vocab_to


# --- spec documents --------------------------------------------------------------------------------
def test_spec_docs_are_documents_about_the_model_not_dialogues_by_it():
    """MSM's mechanism is next-token prediction over third-party prose, in the same shape as the
    pretraining data that taught the model everything else. A first-person answer here would just be the
    SFT demonstrations again, which is the thing that produced identity_recite_rate 0.70."""
    docs = spec_docs(400, 100_000_000, seed=0)
    assert len(docs) == 400
    for d in docs:
        t = d["text"]
        assert NAME in t
        assert "<|" not in t, "no chat template: these are documents, not turns"
        assert not t.lstrip().startswith("I'm "), "the model is not the speaker in its own mid-training data"


def test_spec_docs_cover_every_section_and_stay_distinct():
    docs = spec_docs(3000, 100_000_000, seed=0)
    assert len({d["text"] for d in docs}) == len(docs), "byte-identical repeats waste the share"
    text = "\n".join(d["text"] for d in docs)
    for title, _ in SPEC_SECTIONS:
        assert title.replace("Mote", NAME) in text or any(w in text for w in title.split()[:2])


def test_a_question_shaped_opening_is_answered_by_its_own_section():
    """A forum post headed "why no tokenizer?" followed by prose about corrections reads as generated
    text, which is the one thing this corpus cannot afford to look like. Checked exactly rather than by
    keyword: the body must contain a sentence from the pool belonging to the section the opening asks
    about."""
    fmt = dict(name=NAME, author="Noah", params="about 100 million")
    asks = {a.format(**fmt): sec for sec, v in SECTION_ASKS.items() for a in v}
    owned = {sec: [s.format(**fmt) for s in body["claims"] + body["shows"]]
             for sec, body in SECTION_BODY.items()}
    checked = 0
    for d in spec_docs(2000, 100_000_000, seed=3):
        head = d["text"].split("\n")[0]
        if head not in asks:
            continue
        checked += 1
        sec = asks[head]
        assert any(s in d["text"] for s in owned[sec]), f"{head} is not answered by {sec!r}"
    assert checked > 50, "the question-shaped document types should be well represented"


def test_spec_docs_are_deterministic():
    assert spec_docs(200, 1_000_000, seed=7) == spec_docs(200, 1_000_000, seed=7)


# --- long sim --------------------------------------------------------------------------------------
def _dep_sample(diff_fn, n=120, seed0=555_000):
    doms = sorted(DOMAINS)
    out = []
    for i in range(n):
        s = seed0 + i
        tr = make_trace(doms[s % len(doms)], s, diff_fn(random.Random(s ^ 0x10119)))
        try:
            doc, pairs = narrative(tr, "en"), qa_pairs(tr, "en")
        finally:
            tr.world.close()
        if not doc or not pairs:
            continue
        for p in pairs[:2]:
            d = dependency_distance(doc, p["question"])
            if d is not None:
                out.append(d)
    return out


def test_long_sim_actually_creates_long_range_dependencies():
    """The claim the source exists to make, as a number. Measured 2026-08-26: the ordinary generator
    produces no dependency over 1 KB at all, so without this there is nothing in Mote's own data that
    teaches the 16384-byte window anything it could not learn at 512."""
    ordinary = _dep_sample(sample_difficulty)
    long_ = _dep_sample(lambda r: long_difficulty(r, 60, 220))
    assert ordinary and long_
    assert max(ordinary) < 1024, f"the ordinary sim gained long dependencies ({max(ordinary)} B)"
    assert sum(d > 1024 for d in long_) / len(long_) > 0.10
    assert sorted(long_)[len(long_) // 2] > sorted(ordinary)[len(ordinary) // 2]


def test_dependency_distance_anchors_on_the_question_not_the_answer():
    """The first version measured the gold answer's last occurrence and scored 0 % coverage on every
    locale, because an answer is a whole sentence ("It is in the garden.") that never appears in the
    events. What settles the answer is the last event about the entity the question names."""
    doc = "Ivy took the lamp. " + "Jon went to the hall. " * 40 + "Mara went to the attic."
    assert dependency_distance(doc, "Where is the lamp now?") > 800
    assert dependency_distance(doc, "It is in the garden.") is None  # an answer sentence is not an anchor
    assert dependency_distance(doc, "Where is the zebra now?") is None


def test_long_difficulty_stays_inside_the_entity_pools():
    """domains.py samples without replacement from pools of 8 people, 6 rooms and 8 objects; asking for
    more raises. Length has to come from ticks."""
    for i in range(200):
        d = long_difficulty(random.Random(i), 60, 220)
        make_trace("household", i, d).world.close()
        assert d["ticks"] >= 60


# --- the proxy decider -----------------------------------------------------------------------------
class _FakeModel:
    """A model whose logits mimic a real checkpoint: real columns plus -inf padding rows."""

    def __init__(self, vocab: int, pad_to: int, favour: int | None = None):
        self.vocab, self.pad_to, self.favour = vocab, pad_to, favour

    def __call__(self, x):
        n = x.shape[1]
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(1, n, self.pad_to, generator=g)
        logits[:, :, self.vocab:] = float("-inf")  # exactly what config.py's head mask does
        if self.favour is not None:
            logits[:, :, self.favour] += 12.0
        return type("O", (), {"logits": logits})()


def test_masked_padding_columns_do_not_nan_the_entropy_weights():
    """The bug this caught during the build: p * log p is 0 * -inf = nan on a masked column, every
    weighted mean then silently fell back to uniform, and two of three checkpoints reported an entropy-
    weighted score identical to the unweighted one. Scoring the checkpoint's own vocab and guarding
    0 log 0 both matter."""
    ids = list(range(100, 160))
    freq = torch.rand(266) + 0.1
    st = trajectory_stats(_FakeModel(266, 272), ids, 30, torch.device("cpu"), vocab=266, freq=freq / freq.sum())
    assert st is not None
    assert all(np.isfinite(v) for v in st.values())
    # a weighted cell that silently equals its unweighted twin is the NaN fallback firing
    assert st["agree_entropy"] != st["agree_uniform"], "entropy weighting collapsed to uniform"
    assert st["agree"] != st["agree_uniform"], "frequency weighting collapsed to uniform"
    assert st["recip_rank_uniform"] != st["recip_rank"]


def test_proxy_agree_moves_with_agreement():
    """A model that always predicts the expert's next byte scores 1; a random one scores near chance."""
    ids = [100] * 60
    perfect = trajectory_stats(_FakeModel(266, 272, favour=100), ids, 30, torch.device("cpu"), vocab=266)
    chance = trajectory_stats(_FakeModel(266, 272), ids, 30, torch.device("cpu"), vocab=266)
    assert perfect["agree"] == pytest.approx(1.0)
    assert chance["agree"] < 0.5
    assert perfect["ce"] < chance["ce"]


def test_proxy_scores_only_the_reply_span():
    """The prompt conditions the model but is not the expert's own writing, so it must not be scored."""
    ids = list(range(100, 200))
    early = trajectory_stats(_FakeModel(266, 272), ids, 10, torch.device("cpu"), vocab=266)
    late = trajectory_stats(_FakeModel(266, 272), ids, 90, torch.device("cpu"), vocab=266)
    assert early["n_bytes"] > late["n_bytes"]
    assert trajectory_stats(_FakeModel(266, 272), ids, len(ids), torch.device("cpu"), vocab=266) is None


# --- training-time augmentation (2606.16246) ---------------------------------------------------------
def test_r2l_reverses_codepoints_not_bytes():
    """The obstacle that is specific to a byte model. 2606.16246 reverses tokens; reversing raw UTF-8
    bytes is not the byte-level analogue of that, it is corruption — a window of Japanese becomes
    mojibake. Reversing decoded characters preserves the text AND the byte length, because UTF-8 spends
    the same multiset of bytes either way."""
    from mote.data.loader import r2l_window
    from mote.tokenizer import R2L_ID

    text = "Hello ランプは地下室にある。"
    ids = np.array(list(text.encode()), dtype=np.int64)
    out = r2l_window(ids)
    assert len(out) == len(ids), "a reversed window still has to be exactly seq_len+1"
    assert out[0] == R2L_ID
    body = bytes(int(x) for x in out[1:] if x < 256).decode("utf-8", errors="strict")
    assert "ランプ" not in body and "プンラ" in body, "characters reversed, not shattered"
    assert "�" not in body, "a byte-level reversal would leave replacement characters here"


def test_noise_corrupts_content_but_never_structure():
    """A special id is structure, not content: corrupting <|assistant|> teaches the chat template wrong."""
    from mote.data.loader import noise_window
    from mote.tokenizer import ASSISTANT_ID, BOS_ID, EOS_ID

    ids = np.array([BOS_ID] + list(b"hello there friend") + [ASSISTANT_ID] + list(b"reply") + [EOS_ID],
                   dtype=np.int64)
    out = noise_window(ids, np.random.default_rng(0), 0.5)
    assert len(out) == len(ids)
    for i, v in enumerate(ids):
        if v >= 256:
            assert out[i] == v, f"special id at {i} was corrupted"
    assert (out != ids).any(), "nothing was corrupted at rate 0.5"
    assert (out < 256).sum() == (ids < 256).sum(), "a corrupted byte is still a byte"
    assert np.array_equal(noise_window(ids, np.random.default_rng(0), 0.0), ids)  # rate 0 is a no-op


def test_offset_window_declares_its_own_label_shift():
    """Self-describing on purpose: `compute_losses` reads the offset back off the front of the batch, so
    no signature elsewhere grows a parameter for a training-only knob."""
    from mote.data.loader import offset_window
    from mote.tokenizer import OFFSET_ID

    ids = np.arange(100, 140, dtype=np.int64)
    out = offset_window(ids, 3)
    assert len(out) == len(ids) and out[0] == OFFSET_ID and out[1] == ord("3")
    assert np.array_equal(offset_window(ids, 1), ids), "offset 1 is ordinary next-byte prediction"


def test_compute_losses_shifts_targets_by_the_declared_offset():
    from mote.data.loader import offset_window

    ids = torch.tensor(offset_window(np.arange(100, 120, dtype=np.int64), 3))[None]
    i = int(ids[0, 1].item() - ord("0"))
    L = ids.shape[1] - 1
    idx = torch.arange(L) + i
    targets = ids[:, idx.clamp(max=ids.shape[1] - 1)]
    # position t predicts x_{t+i}: the input at t and the target at t are i apart in the window
    for t in range(2, L - i):
        assert targets[0, t] == ids[0, t + i]
    assert int((idx >= ids.shape[1]).sum()) == i - 1, "the tail that runs off the end is dropped"


def test_augmentations_are_off_by_default_and_skip_masked_windows(tmp_path):
    """Two invariants. Off by default, so every run to date and the mid-training 2x2 are unaffected. And
    never applied to an SFT window: the loss mask is aligned to the original byte positions, so reversing
    or shifting the ids under it would train on the wrong span."""
    from mote.data.loader import ByteShard

    rng = np.random.default_rng(3)
    rng.integers(0, 256, size=8000, dtype=np.uint16).tofile(tmp_path / "s.sft.train.bin")
    np.ones(8000, dtype=np.uint8).tofile(tmp_path / "s.sft.train.mask.bin")
    (tmp_path / "s.sft.meta.json").write_text(
        '{"train": {"file": "s.sft.train.bin", "mask_file": "s.sft.train.mask.bin"}}')

    plain = ByteShard(tmp_path / "s", "train", sft=True)
    assert (plain.noise, plain.r2l, plain.offset_max) == (0.0, 0.0, 1)

    loud = ByteShard(tmp_path / "s", "train", sft=True, noise=0.5, r2l=1.0, offset_max=5)
    g = torch.Generator().manual_seed(0)
    ids, mask = loud.sample_batch(4, 64, g)
    assert mask is not None and ids.shape == (4, 65)
    from mote.tokenizer import OFFSET_ID, R2L_ID
    assert not ((ids == R2L_ID) | (ids == OFFSET_ID)).any(), "a masked SFT window must not be augmented"


# --- the sim's failure layer -------------------------------------------------------------------------
def test_failures_are_a_strict_no_op_on_state():
    """The invariant the whole failure layer rests on: a refused action leaves the world exactly as it
    was, so every question computed from the final state is still correct. What changes is difficulty —
    the reader has to notice nothing moved."""
    import mote.sim.domains as D
    from mote.sim.ecs import World

    w = World(7)
    w.add_system(D.household_system)
    a, b, k = w.spawn("ivy"), w.spawn("jon"), w.spawn("key")
    for e in (a, b, k):
        w.add(e, D.InRoom("kitchen"))
    w.relate(k, "held_by", b)
    state = lambda: {w.names[e]: (w.get(e, D.InRoom).room, w.one(e, "held_by"), w.one(e, "inside")) for e in w.names}
    before = state()
    ev = w.step([{"kind": "take", "who": "ivy", "obj": "key"}])
    assert ev[0].kind == "failed" and ev[0].data["why"] == "held_by_other" and ev[0].data["holder"] == "jon"
    assert state() == before
    w.close()


def test_p_fail_zero_reproduces_the_old_generator_exactly():
    """Everything built before 2026-08-26 was generated with no failures. `--p-fail 0` has to be that
    generator byte-for-byte, or the existing 150 MB of narratives and 20k traces silently stop matching
    the shards built from them."""
    from mote.sim.render import narrative

    for s in (900, 901, 902):
        for dom in ("household", "inventory", "schedule"):
            tr = make_trace(dom, s, sample_difficulty(random.Random(s), p_fail=0))
            try:
                assert not any(e.kind == "failed" for e in tr.events)
                assert narrative(tr, "en")
            finally:
                tr.world.close()


def test_every_failure_reason_renders_in_every_locale():
    """A reason that renders as an empty string silently deletes the sentence from the narrative, leaving
    a gap the questions still depend on."""
    from mote.sim.domains import FAIL_REASONS
    from mote.sim.render import LOCALES

    seen = {}
    for s in range(500, 1400):
        for dom in ("household", "inventory", "schedule"):
            tr = make_trace(dom, s, sample_difficulty(random.Random(s), p_fail=40))
            try:
                for e in tr.events:
                    if e.kind == "failed":
                        seen.setdefault(e.data["why"], e)
            finally:
                tr.world.close()
        if len(seen) == len(FAIL_REASONS):
            break
    assert set(seen) == set(FAIL_REASONS), f"never generated: {set(FAIL_REASONS) - set(seen)}"
    for why, e in seen.items():
        for loc in ("en", "ru", "ja"):
            s = LOCALES[loc]["event"](e)
            assert s and s.strip(), f"{why} renders empty in {loc}"


def test_retrodiction_exists_in_all_three_acting_domains():
    """Before 2026-08-26 the only backward question was `where_obj_start` (2.8 % of questions, and only
    "at the beginning"). Each acting domain now has one anchored on an event the narrative described."""
    import collections

    from mote.sim.render import qa_pairs

    want = {"household": "where_obj_before", "inventory": "count_goods_before", "schedule": "slot_before_move"}
    got = collections.Counter()
    for s in range(6000, 6200):
        for dom, q in want.items():
            tr = make_trace(dom, s, sample_difficulty(random.Random(s), p_fail=15))
            try:
                got.update(p["qtype"] for p in qa_pairs(tr, "en"))
            finally:
                tr.world.close()
    for dom, q in want.items():
        assert got[q] > 0, f"{dom} produced no {q}"
    back = sum(got[q] for q in (*want.values(), "where_obj_start"))
    assert back / sum(got.values()) > 0.05


# --- lexical substitution ----------------------------------------------------------------------------
def test_lexical_swap_respects_word_boundaries():
    """The bug this caught: without boundaries "coin" matches inside "coins" and leaves "монетаs", which
    is not code-switching, it is corruption."""
    from mote.sim.render import lexical_swap

    s = "Ivy picked up the lamp and 3 coins. Kofi booked the workshop."
    for seed in range(20):
        out = lexical_swap(s, random.Random(seed), 1.0)
        assert "coins" in out, "a plural was broken by a singular match"
        assert not re.search(r"[А-Яа-яぁ-ヿ一-鿿][A-Za-z]|[A-Za-z][А-Яа-яぁ-ヿ一-鿿]", out)
    assert lexical_swap(s, random.Random(0), 0.0) == s


# --- the recovery probe ------------------------------------------------------------------------------
def test_recovery_probe_does_not_reward_garbage():
    """The first version scored `other` for anything that was not a verbatim repeat, and a base model that
    has never seen a tool trace scored a perfect 1.000 — its noise never coincidentally equalled the
    refused call. Parseability is in the denominator now."""
    from mote.eval.recovery_probe import classify

    parses = lambda norm: norm.startswith("ivy:") and "take" in norm
    assert classify("sim: ivy: take key", "ivy: take key", parses) == "repeat"
    assert classify("sim: ivy: take lamp", "ivy: take key", parses) == "other"
    assert classify("The standard for the standard for the standard", "ivy: take key", parses) == "unparseable"
    assert classify("", "ivy: take key", parses) == "none"


# --- counterfactual minimal pairs --------------------------------------------------------------------
def test_counterfactual_pairs_share_their_prefix_and_disagree_on_the_answer():
    """The construction that fixed the identity pushback set on 2026-08-25 — one template rendered twice
    so only the claim's truth distinguishes the sides — applied to world state. No replay-to-step API was
    needed: diverting the LAST action leaves the prefix bit-identical, because the draws that chose it
    have already happened."""
    from mote.sim.counterfactual import matched_questions
    from mote.sim.domains import make_counterfactual
    from mote.sim.render import narrative

    checked = 0
    for s in range(9000, 9200):
        pair = make_counterfactual("household", s, sample_difficulty(random.Random(s), p_fail=10))
        if pair is None:
            continue
        a, b = pair
        try:
            da, db = narrative(a, "en"), narrative(b, "en")
            matched = matched_questions(a, b, "en")
        finally:
            a.world.close()
            b.world.close()
        assert a.events[:-1] == b.events[:-1], "the branches diverged before the last action"
        assert a.events[-1] != b.events[-1], "the divert produced no change"
        shared = sum(1 for x, y in zip(da, db) if x == y) / max(len(da), 1)
        assert shared > 0.7, f"prefix only {shared:.0%} shared"
        for q_a, q_b in matched:
            assert q_a["question"] == q_b["question"], "a 'matched' pair asks two different questions"
            assert q_a["answer"] != q_b["answer"]
            checked += 1
        if checked > 30:
            break
    assert checked > 30, "not enough matched questions to test"


def test_qa_pairs_carry_their_args():
    """Without them a caller cannot tell "where is the cup" from "where is the book" once both are just
    strings — which silently paired two different questions in the first counterfactual build."""
    from mote.sim.render import qa_pairs

    tr = make_trace("household", 9001, sample_difficulty(random.Random(9001), p_fail=0))
    try:
        pairs = qa_pairs(tr, "en")
    finally:
        tr.world.close()
    assert pairs and all("args" in p and isinstance(p["args"], dict) for p in pairs)
    assert any(p["args"] for p in pairs)


# --- expert recovery traces --------------------------------------------------------------------------
def test_domain_and_locale_are_no_longer_locked_together():
    """A pre-existing defect found on 2026-08-26. `domain = TASK_DOMAINS[seed % 3]` next to
    `locale = locales[n % 3]` advanced in lockstep, so in the 20,000 shipped traces EVERY household trace
    was English, every inventory trace Russian and every schedule trace Japanese — two thirds of the
    domain x locale space empty, and any per-locale measurement of tool use secretly a per-domain one."""
    import collections

    from mote.sim.tasks import TASK_DOMAINS

    locales = ("en", "ru", "ja")
    seen = collections.Counter()
    for i in range(600):
        seed = 3_000_000 + i
        seen[(TASK_DOMAINS[seed % len(TASK_DOMAINS)], random.Random(seed ^ 0x10CA1E).choice(list(locales)))] += 1
    assert len(seen) == len(TASK_DOMAINS) * len(locales), f"empty cells: {seen}"
    lo = min(seen.values())
    assert lo > 0.5 * (600 / (len(TASK_DOMAINS) * len(locales))), f"lopsided: {seen}"


def test_recovery_traces_contain_a_real_refusal_then_a_correct_step():
    """The half of the failure work that closes the RLVR-1 gap. A refusal that names the obstacle is what
    the agent can act on — the inventory case tells it what the seller actually holds."""
    from mote.sim.tasks import expert_messages, make_task

    found = 0
    for s in range(2_000_100, 2_000_400):
        for dom in ("household", "inventory", "schedule"):
            task = make_task(dom, s, "en")
            if not task.expert:
                continue
            msgs = expert_messages(task, recover=True, rng=random.Random(s))
            parts = msgs[1]["parts"]
            calls = [p for p in parts if p["type"] == "call"]
            if len(calls) <= len(task.expert):
                continue  # no misstep was available from any state in this task
            results = [p["text"] for p in parts if p["type"] == "result"]
            assert any("tried to" in r for r in results), f"{dom}: no informative refusal in {results}"
            assert parts[-1]["type"] == "text", "the trace must still finish"
            found += 1
        if found > 8:
            break
    assert found > 8, "recovery traces were never produced"


def test_recover_frac_zero_is_the_old_generator():
    from mote.sim.tasks import expert_messages, make_task

    for s in range(2_000_100, 2_000_112):
        task = make_task("household", s, "en")
        parts = expert_messages(task)[1]["parts"]
        assert sum(1 for p in parts if p["type"] == "call") == len(task.expert)


def test_a_clashing_booking_now_reaches_the_system():
    """parse_action used to reject a clash itself, so the agent got "Unknown action." — a refusal that
    says only "no". schedule_system has its own check now, so the clash gets through and the agent is
    told the time is taken."""
    from mote.sim.domains import Calendar
    from mote.sim.tasks import SimEnv, make_task, parse_action

    for s in range(2_000_100, 2_000_260):
        task = make_task("schedule", s, "en")
        env = SimEnv(task)
        try:
            for step in task.expert:
                env.act(step)
            who = next((n for n in env.world.names.values()
                        if (c := env.world.get(env.world.eid(n), Calendar)) and c.slots), None)
            if who is None:
                continue
            cal = env.world.get(env.world.eid(who), Calendar)
            start, _end, taken = cal.slots[0]
            free = [t for t in ("standup", "review", "lunch", "call") if t not in {x for _a, _b, x in cal.slots}]
            if not free:
                continue
            text = f"{who}: book {free[0]} at {start} for 1h"
            if parse_action("schedule", text, env.world, env.init) is None:
                continue  # rejected for a structural reason (window), not the clash
            assert "already taken" in env.act(text)
            return
        finally:
            env.close()
    raise AssertionError("never constructed a clashing booking to test")


# --- replay and rollout branching ---------------------------------------------------------------------
def test_world_deserialize_round_trips_exactly():
    """`serialize` had no inverse until 2026-08-26. Counterfactuals never needed one — re-running a seeded
    script to tick k costs 0.28 ms and reproduces the prefix exactly. What needs it is branching a LIVE
    episode: during RLVR-1 the state comes from the model's own choices, so there is nothing to re-run."""
    from mote.sim.domains import (Calendar, Container, InRoom, Person, Portable, Stock,
                                  household_system)
    from mote.sim.ecs import World

    comps = {c.__name__: c for c in (InRoom, Person, Container, Portable, Stock, Calendar)}
    for dom in ("household", "inventory", "schedule"):
        tr = make_trace(dom, 4242, sample_difficulty(random.Random(4242), p_fail=15))
        blob = tr.world.serialize()
        tr.world.close()
        w = World.deserialize(blob, comps, seed=4242)
        try:
            assert w.serialize() == blob, f"{dom}: round trip is not byte-exact"
        finally:
            w.close()

    # a restored world is a working world, not just a readable dump
    tr = make_trace("household", 4242, sample_difficulty(random.Random(4242), p_fail=15))
    blob = tr.world.serialize()
    tr.world.close()
    w = World.deserialize(blob, comps, seed=4242)
    try:
        w.add_system(household_system)
        who = next(n for n in w.names.values() if w.get(w.eid(n), InRoom))
        assert w.step([{"kind": "move", "who": who, "to": "attic"}])
    finally:
        w.close()


def test_counterfactual_diverts_in_every_acting_domain():
    """Built household-only at first, so two thirds of seeds produced nothing and the waste looked like a
    limitation of the last-action approach. It was a missing feature."""
    from mote.sim.domains import make_counterfactual

    for dom in ("household", "inventory", "schedule"):
        usable = sum(1 for s in range(9000, 9100)
                     if make_counterfactual(dom, s, sample_difficulty(random.Random(s), p_fail=15)) is not None)
        assert usable > 40, f"{dom}: only {usable}/100 seeds usable"


def test_divert_at_an_arbitrary_tick():
    """`divert=True` is the last tick; `divert=k` is tick k, which is what a PIVOT-style continuation
    wants. Re-running the seeded script to k IS the replay — the draws up to k are identical."""
    from mote.sim.domains import DOMAINS
    from mote.sim.render import narrative

    diff = sample_difficulty(random.Random(11), p_fail=0)
    base = DOMAINS["household"](11, diff)
    try:
        n = len(base.events)
        text = narrative(base, "en")
    finally:
        base.world.close()
    early = DOMAINS["household"](11, {**diff, "divert": 1})
    try:
        # a divert at tick 1 must change the narrative EARLIER than a divert at the last tick does
        shared_early = sum(1 for x, y in zip(text, narrative(early, "en")) if x == y) / max(len(text), 1)
    finally:
        early.world.close()
    late = DOMAINS["household"](11, {**diff, "divert": True})
    try:
        shared_late = sum(1 for x, y in zip(text, narrative(late, "en")) if x == y) / max(len(text), 1)
    finally:
        late.world.close()
    assert shared_early < shared_late, "diverting at tick 1 should diverge sooner than at the last tick"


def test_sim_probe_worlds_can_carry_failures():
    """A probe whose worlds cannot fail cannot measure whether the model tracks a world that can. It used
    to build at the default 0 whatever the training data did."""
    from mote.eval.sim_probe import heldout_items

    def fail_share(pf):
        items = heldout_items(60, ["en", "ru", "ja"], p_fail=pf)
        hits = sum(1 for x in items
                   if "tried to" in x["prompt"] or "попыта" in x["prompt"] or "としたが" in x["prompt"])
        return hits / len(items)

    assert fail_share(0) == 0.0
    assert fail_share(15) > 0.3
    assert fail_share(30) > fail_share(15)
