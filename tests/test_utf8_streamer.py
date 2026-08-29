"""The byte-at-a-time decoder the Studio streams through: never raises, reproduces valid text exactly under any
split, and emits only valid strings on garbage."""

import random

from mote.tokenizer import BYTE_VOCAB, EOS_ID, Utf8Streamer


def _stream(ids):
    s, out = Utf8Streamer(), []
    for i in ids:
        out.append(s.feed(i))
    return "".join(out), s.pending


def test_valid_text_is_reproduced_exactly_byte_by_byte():
    text = "The router — 路由器 — draws a boundary. ランプ 🙂 émoji, naïve, Ω≈ω."
    out, pending = _stream(list(text.encode("utf-8")) + [EOS_ID])
    assert out == text and pending == b""


def test_garbage_never_raises_and_yields_valid_text():
    rng = random.Random(0)
    for _ in range(200):
        ids = [rng.randrange(0, BYTE_VOCAB + 8) for _ in range(rng.randrange(1, 64))]
        out, pending = _stream(ids)
        assert isinstance(out, str)
        out.encode("utf-8")  # valid unicode
        assert len(pending) <= 3  # at most an incomplete character is held back
