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

import numpy as np
import pytest
import torch

from mote.data.build_spec_docs import SECTION_ASKS, SECTION_BODY, generate as spec_docs
from mote.data.loader import fim_window
from mote.eval.proxy import trajectory_stats
from mote.serve.identity import NAME, SPEC_SECTIONS
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
