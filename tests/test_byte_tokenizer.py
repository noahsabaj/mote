import pytest
from sisa_mamba.tokenizer.byte_tokenizer import ByteTokenizer


def test_byte_tokenizer_lossless_roundtrip():
    """Verifies lossless UTF-8 encoding and decoding for arbitrary text, emojis, and machine bytes."""
    tok = ByteTokenizer()

    test_strings = [
        "Hello, World!",
        "Mathematical sequence modeling: SISA × Mamba-3 ∞",
        "Machine code: \x00\x01\x02\xff\xfe\xfd",
        "Python, Rust, CUDA, Triton, CuTe 🚀🔥⚡",
        "1 + 1 = 2 ⇒ e^(i*pi) + 1 = 0",
    ]

    for s in test_strings:
        tokens = tok.encode(s, add_bos=True, add_eos=True)
        assert tokens[0] == tok.bos_id
        assert tokens[-1] == tok.eos_id

        decoded = tok.decode(tokens, skip_special_tokens=True)
        assert decoded == s


def test_incremental_streaming_decoder():
    """Verifies that UTF-8 multi-byte characters are decoded properly in a real-time stream."""
    tok = ByteTokenizer()
    streamer = tok.get_incremental_decoder()

    # Multi-byte emoji: 🚀 (F0 9F 98 80)
    emoji_str = "🚀"
    raw_bytes = list(emoji_str.encode("utf-8"))

    emitted = ""
    for b in raw_bytes:
        emitted += streamer.step(b)

    assert emitted == emoji_str
