"""The optional access token gates /api, /v1 and /ws/generate; /api/health stays open."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import mote.serve.app as A


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(A.STUDIO, "token", "s3cret")
    monkeypatch.setattr(A.STUDIO, "engine", None)
    return TestClient(A.app)


def test_http_routes_require_bearer_token(client):
    assert client.get("/api/health").status_code == 200
    r = client.get("/api/model")
    assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "Bearer"
    assert client.get("/api/model", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 401
    # right token: past the gate, and 503 because no engine is loaded in this test
    assert client.get("/api/model", headers={"Authorization": "Bearer s3cret"}).status_code == 503


def test_no_token_means_open(monkeypatch):
    monkeypatch.setattr(A.STUDIO, "token", None)
    monkeypatch.setattr(A.STUDIO, "engine", None)
    assert TestClient(A.app).get("/api/model").status_code == 503
    with TestClient(A.app).websocket_connect("/ws/generate") as ws:
        ws.send_json({"type": "auth", "token": "anything"})  # answered even when no token is configured
        assert ws.receive_json() == {"type": "auth_ok"}


def test_pairing_page_is_loopback_only_and_codes_work_once(monkeypatch):
    monkeypatch.setattr(A.STUDIO, "token", "s3cret")
    monkeypatch.setattr(A.STUDIO, "engine", None)
    A.PAIRING.public_url = "https://example.ts.net"
    remote = TestClient(A.app, client=("10.0.0.7", 1234))
    assert remote.get("/pair").status_code == 403
    assert remote.get("/pair/code").status_code == 403
    local = TestClient(A.app, client=("127.0.0.1", 1234))
    page = local.get("/pair")
    assert page.status_code == 200 and "https://example.ts.net/#token=s3cret" in page.text
    code = local.get("/pair/code").json()["code"]
    assert len(code) == 6 and code.isdigit()
    assert remote.post("/api/pair", json={"code": "000000" if code != "000000" else "000001"}).status_code == 400
    assert remote.post("/api/pair", json={"code": code}).json() == {"token": "s3cret"}
    assert remote.post("/api/pair", json={"code": code}).status_code == 400  # single use


def test_pairing_is_rate_limited(monkeypatch):
    monkeypatch.setattr(A.STUDIO, "token", "s3cret")
    A.PAIRING._attempts.clear()
    c = TestClient(A.app, client=("10.0.0.7", 1234))
    codes = [c.post("/api/pair", json={"code": "999999"}).status_code for _ in range(11)]
    assert codes[:10] == [400] * 10 and codes[10] == 429


def test_websocket_needs_auth_frame_first(client):
    with client.websocket_connect("/ws/generate") as ws:
        ws.send_json({"type": "generate", "messages": []})
        with pytest.raises(WebSocketDisconnect) as ei:
            ws.receive_json()
        assert ei.value.code == 4401
    with client.websocket_connect("/ws/generate") as ws:
        ws.send_json({"type": "auth", "token": "nope"})
        with pytest.raises(WebSocketDisconnect) as ei:
            ws.receive_json()
        assert ei.value.code == 4401
    with client.websocket_connect("/ws/generate") as ws:
        ws.send_json({"type": "auth", "token": "s3cret"})
        assert ws.receive_json() == {"type": "auth_ok"}
        ws.send_json({"type": "generate", "messages": []})
        assert ws.receive_json()["type"] == "error"  # no engine loaded → error event, not a 4401
