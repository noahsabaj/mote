"""Tokenizer (chat template, UTF-8 streaming) and serving-engine event-stream checks on a tiny fresh model."""

import threading

import pytest
import torch

from morpheme.config import MBPCfg, Mamba3Cfg, MorphemeConfig, RelationCfg
from morpheme.model.hnet import HNetForCausalLM
from morpheme.serve.engine import Engine, GenParams
from morpheme.tokenizer import ASSISTANT_ID, BOS_ID, EOS_ID, USER_ID, ByteTokenizer, ChatMessage, Utf8Streamer


def test_chat_template_and_loss_mask():
    tok = ByteTokenizer()
    msgs = [ChatMessage("user", "hi"), ChatMessage("assistant", "yo")]
    ids = tok.format_chat(msgs)
    assert ids == [BOS_ID, USER_ID, ord("h"), ord("i"), EOS_ID, ASSISTANT_ID, ord("y"), ord("o"), EOS_ID, ASSISTANT_ID]
    ids2, mask = tok.format_chat_with_loss_mask(msgs)
    assert ids2 == ids[:-1]
    assert mask == [0, 0, 0, 0, 0, 0, 1, 1, 1]  # assistant bytes + their EOS only
    assert tok.decode(ids, skip_special_tokens=True) == "hiyo"


def test_utf8_streamer_assembles_multibyte_and_survives_garbage():
    s = Utf8Streamer()
    out = ""
    for b in "é—z".encode("utf-8"):
        out += s.feed(b)
    assert out == "é—z" and s.pending == b""
    # a stray continuation byte must not wedge the decoder
    assert s.feed(0x80) == ""
    assert s.pending == b""
    assert s.feed(ord("a")) == "a"
    # pending bytes are reported while a character is incomplete
    assert s.feed(0xE2) == "" and s.pending == b"\xe2"


def _tiny_engine(tmp_path):
    cfg = MorphemeConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    run = tmp_path / "runs" / "pilot_tiny"
    run.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 3, "config": cfg.to_dict(), "extra": {"bytes_seen": 3000}}, run / "last.pt")
    return Engine(run / "last.pt", device="cpu")


@pytest.fixture
def no_stop_ids(monkeypatch):
    # a random-init model samples EOS/role ids almost immediately; neutralize them so the stream has length
    import morpheme.serve.engine as E

    monkeypatch.setattr(E, "STOP_IDS", set())


def test_engine_event_stream_contract(tmp_path, no_stop_ids):
    eng = _tiny_engine(tmp_path)
    info = eng.info()
    assert info["status"] == "pilot" and info["context_limit_bytes"] == 256 and info["params"] > 0
    events = []
    eng.generate([{"role": "user", "content": "hi"}], GenParams(temperature=0.9, top_p=0.95, max_bytes=24, n_candidates=3), events.append, threading.Event())
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "done"
    bytes_ev = [e for e in events if e["type"] == "byte"]
    assert 1 <= len(bytes_ev) <= 24
    assert bytes_ev[0]["i"] == 0 and all(e["i"] == i for i, e in enumerate(bytes_ev))
    for e in bytes_ev:
        assert set(e) >= {"byte", "text", "pending", "p", "entropy", "boundary", "boundary_p", "chunk", "source", "t_ms"}
        assert 0 <= e["byte"] < 262 and 0.0 <= e["p"] <= 1.0 + 1e-6 and e["source"] in ("nbp", "mbp", "fix")
    done = events[-1]
    assert done["reason"] == "max_bytes" and len(bytes_ev) == 24
    assert done["stats"]["bytes"] == len(bytes_ev)
    # a boundary early in the reply triggers a draft; every draft byte is either accepted (mbp) or corrected (fix)
    if any(e["boundary"] for e in bytes_ev[:-3]):
        assert done["stats"]["mbp_proposed"] > 0 and done["stats"]["spec_rounds"] > 0
        assert any(e["source"] in ("mbp", "fix") for e in bytes_ev)
        assert done["stats"]["mbp_accepted"] + done["stats"]["spec_fixes"] >= 1
    assert any(e["type"] == "stats" for e in events)
    if any(e["boundary"] for e in bytes_ev):  # diagnostics are emitted at chunk boundaries
        assert any(e["type"] == "diagnostics" for e in events)


def test_engine_stop_flag(tmp_path, no_stop_ids):
    eng = _tiny_engine(tmp_path)
    stop = threading.Event()
    events = []

    def emit(e):
        events.append(e)
        if e["type"] == "byte" and e["i"] == 2:
            stop.set()

    eng.generate([{"role": "user", "content": "hi"}], GenParams(max_bytes=100, n_candidates=0), emit, stop)
    assert events[-1]["type"] == "done" and events[-1]["reason"] == "stopped"
    assert len([e for e in events if e["type"] == "byte"]) == 3
