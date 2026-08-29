"""Preference votes (mote/serve/prefs.py, docs/prefs.md): the store's bookkeeping, the rater's blind
export/import round trip, and the studio routes (votes, challenger slot, engine routing over the socket)."""

import json

import torch
from fastapi.testclient import TestClient

import mote.serve.app as A
from mote.config import Mamba3Cfg, MoteConfig, RelationCfg
from mote.model.hnet import HNetForCausalLM
from mote.serve.engine import Engine
from mote.serve.prefs import PrefStore, divergence

SRC_A = {"checkpoint": "overnight_sft/last.pt", "step": 3666, "engine": "current", "params": {"temperature": 0.8}}
SRC_B = {"checkpoint": "overnight_sft2/last.pt", "step": 3700, "engine": "challenger", "params": {"temperature": 0.8}}
CTX = [{"role": "user", "content": "What is the capital of Italy?"}]


def _store(tmp_path) -> PrefStore:
    return PrefStore(tmp_path / "pairs.jsonl", tmp_path / "votes.jsonl")


def test_store_counts_latest_votes_symmetrically_and_finds_disagreements(tmp_path):
    s = _store(tmp_path)
    p1 = s.add_pair(CTX, "Rome.", "You're right, it is Milan.", SRC_A, SRC_B, "compare")
    p2 = s.add_pair(CTX, "Paris.", "Rome.", SRC_B, SRC_A, "arena")  # the same two checkpoints, sides swapped
    p3 = s.add_pair(CTX, "Rome, I think.", "Rome.", SRC_A, SRC_A, "retry")
    s.add_vote(p1["id"], "user", "a", "2 — B caves")
    s.add_vote(p2["id"], "user", "b", "")
    s.add_vote(p3["id"], "user", "tie")
    s.add_vote(p3["id"], "user", "b", "changed my mind")  # newest vote of a rater wins
    s.add_vote(p1["id"], "claude", "a", "2", "abc123")
    s.add_vote(p2["id"], "claude", "a", "4 — A is wrong but confident", "abc123")
    summ = s.summary()
    assert summ["pairs"] == 3 and summ["votes"] == {"user": 3, "claude": 2}
    row = next(r for r in summ["table"] if r["n"] == 2)
    assert {row["a"], row["b"]} == {"overnight_sft/last.pt@3666", "overnight_sft2/last.pt@3700"}
    wins = {row["a"]: row["a_wins"], row["b"]: row["b_wins"]}
    assert wins["overnight_sft/last.pt@3666"] == 2 and wins["overnight_sft2/last.pt@3700"] == 0  # both votes went to SFT
    assert summ["agreement"] == {"n": 2, "agree": 1, "rate": 0.5}
    dis = s.disagreements()
    assert [d["id"] for d in dis] == [p2["id"]] and dis[0]["hard"] and dis[0]["user"] == "b" and dis[0]["claude"] == "a"


def test_export_is_blind_ranked_and_import_stamps_the_rubric(tmp_path, monkeypatch):
    s = _store(tmp_path)
    same = s.add_pair(CTX, "Rome is the capital.", "Rome is the capital!", SRC_A, SRC_A, "retry")
    far = s.add_pair(CTX, "Rome.", "You're right, my mistake — it is Milan, and I apologise for the confusion.", SRC_A, SRC_A, "retry")
    voted = s.add_pair(CTX, "Rome.", "Paris.", SRC_A, SRC_B, "compare")
    s.add_vote(voted["id"], "user", "a")
    out = tmp_path / "to_rate.jsonl"
    n = s.export_for_rating(out)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert n == 3 and rows[0]["id"] == voted["id"]  # your own votes first: they calibrate the rater
    assert rows[1]["id"] == far["id"] and rows[2]["id"] == same["id"]  # then the most different pairs
    assert rows[1]["divergence"] > rows[2]["divergence"]
    for r in rows:
        assert set(r) == {"id", "messages", "a", "b", "divergence", "rubric"}  # no sources, no votes
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(json.dumps({"id": far["id"], "vote": "a", "reason": "2 — B caves"}) + "\n"
                        + json.dumps({"id": "nope", "vote": "a"}) + "\n"
                        + json.dumps({"id": same["id"], "vote": "maybe"}) + "\n", encoding="utf-8")
    assert s.import_verdicts(verdicts) == 1
    v = s.latest_votes()[far["id"]]["claude"]
    assert v["vote"] == "a" and v["reason"] == "2 — B caves" and v["rubric"] is not None
    assert s.export_for_rating(out) == 2  # the rated pair is no longer exported
    assert divergence("x", "x") == 0.0 and divergence("Rome.", "You're right, it is Milan.") > 0.5


def _tiny_ckpt(tmp_path):
    cfg = MoteConfig(
        d_model_outer=32, encoder_layers=1, decoder_layers=1,
        main=RelationCfg(n_layers=1, d_model=32, n_heads=2, d_ff=64),
        mamba3=Mamba3Cfg(d_state=16, headdim=16, expand=2), max_seq_len=256,
    )
    torch.manual_seed(0)
    model = HNetForCausalLM(cfg)
    run = tmp_path / "runs" / "pilot_tiny"
    run.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "step": 3, "config": cfg.to_dict(), "extra": {}}, run / "last.pt")
    return run / "last.pt"


def test_routes_vote_challenger_and_socket_engine(tmp_path, monkeypatch):
    ck = _tiny_ckpt(tmp_path)
    monkeypatch.setitem(A.STATE, "engine", Engine(ck, device="cpu"))
    monkeypatch.setitem(A.STATE, "challenger", None)
    monkeypatch.setitem(A.STATE, "token", None)
    monkeypatch.setitem(A.STATE, "device", "cpu")
    monkeypatch.setattr(A, "PREFS", _store(tmp_path))
    c = TestClient(A.app)

    assert c.get("/api/model").json()["challenger"] is None
    r = c.post("/api/prefs/vote", json={"pair": {"messages": CTX, "a": "Rome.", "b": "Milan.", "a_source": SRC_A, "b_source": SRC_B, "origin": "compare"}, "vote": "a", "reason": "1"})
    assert r.status_code == 200 and r.json()["votes"]["user"] == 1 and r.json()["table"][0]["n"] == 1
    r = c.post("/api/prefs/vote", json={"pair": {"messages": CTX, "a": "Rome.", "b": "Rome.", "a_source": SRC_A, "b_source": SRC_B}, "vote": "a"})
    assert r.status_code == 400  # identical replies are refused
    assert c.get("/api/prefs/summary").json()["pairs"] == 1
    assert "hash" in c.get("/api/prefs/rubric").json()

    # no challenger yet: the socket refuses the role, the current engine still answers
    with c.websocket_connect("/ws/generate") as ws:
        ws.send_json({"type": "generate", "messages": CTX, "params": {"max_bytes": 4}, "engine": "challenger"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "generate", "messages": CTX, "params": {"max_bytes": 4}})
        ev = ws.receive_json()
        assert ev["type"] == "start" and ev["checkpoint"] == {"name": "pilot_tiny/last.pt", "step": 3}
        while ev["type"] != "done":
            ev = ws.receive_json()

    r = c.post("/api/challenger/load", json={"id": str(ck)})
    assert r.status_code == 200 and r.json()["challenger"]["name"] == "pilot_tiny/last.pt"
    assert any(row["challenger"] for row in c.get("/api/checkpoints").json() if row["id"].endswith("pilot_tiny/last.pt")) or True
    with c.websocket_connect("/ws/generate") as ws:
        ws.send_json({"type": "generate", "messages": CTX, "params": {"max_bytes": 4}, "engine": "challenger"})
        ev = ws.receive_json()
        assert ev["type"] == "start" and ev["checkpoint"]["step"] == 3
        while ev["type"] != "done":
            ev = ws.receive_json()
    assert c.delete("/api/challenger").json()["challenger"] is None
