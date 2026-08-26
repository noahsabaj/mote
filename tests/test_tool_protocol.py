"""The tool protocol (docs/shape.md § pipeline, docs/search.md): <|call|> name: args <|result|> inside an
assistant turn — rendered and masked by the tokenizer, routed by the engine's hook, the result injected
into the running state, decoding resumed; the call bytes never become reply text."""

import threading

import pytest
import torch

from mote.config import MBPCfg, Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM
from mote.serve.engine import Engine, GenParams
from mote.tokenizer import ASSISTANT_ID, BOS_ID, CALL_ID, EOS_ID, RESULT_ID, USER_ID, VOCAB_SIZE, ByteTokenizer, ChatMessage, parse_call


def test_parse_call_and_vocab():
    # 266 until 2026-08-26, when the three FIM sentinels for the mid-training tool traces took 266-268
    # (2607.12463). The ids below are frozen: they are baked into every existing checkpoint and every
    # built shard, so a renumbering silently reinterprets stored bytes.
    assert VOCAB_SIZE == 269 and CALL_ID == 262 and RESULT_ID == 263
    assert parse_call("search: byte-level tokenizers") == ("search", "byte-level tokenizers")
    assert parse_call(" Sim : take candle ") == ("sim", "take candle") and parse_call("nothing") == ("nothing", "")


def test_tool_turn_rendering_and_mask():
    tok = ByteTokenizer()
    msgs = [ChatMessage("user", "q"),
            ChatMessage("assistant", "", parts=[{"type": "call", "text": "sim: go"}, {"type": "result", "text": "ok"}, {"type": "text", "text": "done"}])]
    ids, mask = tok.format_chat_with_loss_mask(msgs)
    exp = [BOS_ID, USER_ID, ord("q"), EOS_ID, ASSISTANT_ID, CALL_ID, *b"sim: go", RESULT_ID, *b"ok", ASSISTANT_ID, *b"done", EOS_ID]
    assert ids == exp
    # the call (both markers + its bytes) and the answer train; the server-written result and <|assistant|> do not
    assert mask == [0, 0, 0, 0, 0] + [1] * 9 + [0, 0, 0] + [1] * 4 + [1]
    assert tok.format_chat(msgs) == ids + [ASSISTANT_ID]
    assert tok.format_chat_with_loss_mask([ChatMessage("user", "hi"), ChatMessage("assistant", "yo")])[1] == [0, 0, 0, 0, 0, 0, 1, 1, 1]  # unchanged
    assert tok.decode(ids, skip_special_tokens=False).count("<|call|>") == 1


def _tiny_engine(tmp_path):
    cfg = MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mbp=MBPCfg(n_layers=1, n_heads=2, d_ff=64, n_candidates=3, enabled=False),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    run = tmp_path / "runs" / "tiny"
    run.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 1, "config": cfg.to_dict(), "extra": {}}, run / "last.pt")
    return Engine(run / "last.pt", device="cpu")


@pytest.fixture
def only_result_stops(monkeypatch):
    # a random model samples EOS/role ids at once; keep only <|result|> as a stop so the stream has length
    import mote.serve.engine as E
    monkeypatch.setattr(E, "STOP_IDS", {RESULT_ID})


def _run(eng, params):
    ev = []
    eng.generate([{"role": "user", "content": "hello"}], params, ev.append, threading.Event(), context={"want_ids": True})
    return ev


def test_tool_hook_round_trip(tmp_path, only_result_stops):
    eng = _tiny_engine(tmp_path)
    seen = []
    eng.register_tool("echo", lambda a: (seen.append(a), f"pong:{a}")[1])
    script = [ord("A"), CALL_ID, *b"echo: hi", RESULT_ID] + [ord("b")] * 21  # 10 + 9 injected + 21 = the 40-byte budget
    ev = _run(eng, GenParams(temperature=0.0, top_p=1.0, max_bytes=40, n_candidates=0, script=list(script)))
    tools = [e for e in ev if e["type"] == "tool"]
    done = ev[-1]
    assert seen == ["hi"] and len(tools) == 1
    assert tools[0]["tool"] == "echo" and tools[0]["args"] == "hi" and tools[0]["result"] == "pong:hi" and tools[0]["index"] == 1
    bytes_ev = [e for e in ev if e["type"] == "byte"]
    assert [e["source"] for e in bytes_ev[:10]] == ["nbp"] + ["call"] * 9  # 'A', then <|call|> + 8 call bytes
    assert all(e["text"] is None for e in bytes_ev[1:10]) and bytes_ev[0]["text"] == "A"
    assert done["type"] == "done" and done["calls"] == 1 and "echo" not in done["text"] and done["text"].startswith("A")
    # decoding resumed after the injected result and ran to the byte budget (result ids count toward it)
    i_tool = ev.index(tools[0])
    after = [e for e in ev[i_tool + 1:] if e["type"] == "byte"]
    assert len(after) == 21 and all(e["source"] == "nbp" for e in after) and done["stats"]["bytes"] == 40
    assert done["text"] == "A" + "b" * 21 and done["reason"] == "max_bytes"
    # the RL view: every generated id with a loss mask that is 0 exactly on the injected <|result|>pong:hi<|assistant|>
    assert len(done["ids"]) == 40 == len(done["mask"]) and done["mask"].count(0) == 9 and done["mask"][10:19] == [0] * 9
    assert done["ids"][10] == RESULT_ID and done["ids"][18] == ASSISTANT_ID and bytes(done["ids"][11:18]) == b"pong:hi" and done["eos"] is False


def test_unknown_tool_and_call_cap(tmp_path, only_result_stops):
    eng = _tiny_engine(tmp_path)
    script = [CALL_ID, *b"nope: x", RESULT_ID, CALL_ID, *b"nope: y", RESULT_ID]
    ev = _run(eng, GenParams(temperature=0.0, top_p=1.0, max_bytes=60, n_candidates=0, max_calls=1, script=list(script)))
    tools = [e for e in ev if e["type"] == "tool"]
    assert len(tools) == 1 and tools[0]["result"] == "(no such tool: nope)" and tools[0]["args"] == "x"
    assert ev[-1]["reason"] == "eos" and ev[-1]["calls"] == 1  # the second <|result|> ends the reply: the cap is 1

    ev = _run(eng, GenParams(temperature=0.0, top_p=1.0, max_bytes=30, n_candidates=0, max_calls=0, script=[CALL_ID, *b"nope: x", RESULT_ID]))
    assert not [e for e in ev if e["type"] == "tool"] and ev[-1]["reason"] == "eos" and ev[-1]["calls"] == 0


def test_result_without_call_is_a_plain_stop(tmp_path, only_result_stops):
    eng = _tiny_engine(tmp_path)
    eng.register_tool("echo", lambda a: "never")
    ev = _run(eng, GenParams(temperature=0.0, top_p=1.0, max_bytes=30, n_candidates=0, script=[ord("x"), RESULT_ID]))
    assert ev[-1]["reason"] == "eos" and ev[-1]["calls"] == 0 and ev[-1]["text"] == "x"
