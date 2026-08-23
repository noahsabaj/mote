"""The optional access token gates /api, /v1 and /ws/generate; /api/health stays open."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import morpheme.serve.app as A


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(A.STATE, "token", "s3cret")
    monkeypatch.setitem(A.STATE, "engine", None)
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
    monkeypatch.setitem(A.STATE, "token", None)
    monkeypatch.setitem(A.STATE, "engine", None)
    assert TestClient(A.app).get("/api/model").status_code == 503
    with TestClient(A.app).websocket_connect("/ws/generate") as ws:
        ws.send_json({"type": "auth", "token": "anything"})  # answered even when no token is configured
        assert ws.receive_json() == {"type": "auth_ok"}


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
