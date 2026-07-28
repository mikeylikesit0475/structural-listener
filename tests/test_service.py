"""Sprint 3.1: FastAPI service endpoints + WebSocket stream.

Uses the FastAPI test client (httpx-based). Tests the inject endpoint, the WebSocket
stream (meta, waveforms, hits, features, localisation messages), and the error
case for an unknown session.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from structural_listener.service.app import app

client = TestClient(app)


def test_health_check() -> None:
    """GET / returns 200 and a status."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_inject_returns_session_and_coords() -> None:
    """POST /inject returns a session id, source coords, and sensor positions."""
    r = client.post("/inject", params={"source_type": "crack", "x": 0.5, "y": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert "session_id" in body
    assert body["source_xy_m"] == [0.5, 0.5]
    assert body["source_type"] == "crack"
    assert len(body["sensor_positions_m"]) == 4
    assert body["sample_rate_hz"] > 0


def test_websocket_streams_event_messages() -> None:
    """The WebSocket sends meta, 4 waveforms, hits, features, localisation."""
    r = client.post("/inject", params={"source_type": "crack", "x": 0.3, "y": 0.7})
    session_id = r.json()["session_id"]
    with client.websocket_connect(f"/stream/{session_id}") as ws:
        types: list[str] = []
        # meta + 4 waveforms + hits + features + localisation = 8 messages
        for _ in range(8):
            msg = json.loads(ws.receive_text())
            types.append(msg["type"])
        assert "meta" in types
        assert types.count("waveform") == 4
        assert "hits" in types
        assert "localisation" in types


def test_websocket_unknown_session_sends_error() -> None:
    """An unknown session id gets an error message and the socket closes."""
    with client.websocket_connect("/stream/nonexistent") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
