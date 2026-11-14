"""WebSocket protocol: streaming, resume, end-of-session summary, input validation."""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import create_app


def _recv_until(ws, kind):
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] == kind:
            return events


def test_stream_a_reply_then_end_the_session(container):
    with TestClient(create_app(container)) as client:
        with client.websocket_connect("/ws/chat?user_id=u1&session_id=sess1") as ws:
            assert ws.receive_json() == {"type": "session", "session_id": "sess1"}

            ws.send_json({"type": "message", "text": "I prefer advice in Hindi."})
            events = _recv_until(ws, "done")
            tokens = [e["text"] for e in events if e["type"] == "token"]
            assert len(tokens) > 1                                     # streamed, not one blob
            assert "".join(tokens).strip() == events[-1]["response"]

            ws.send_json({"type": "end"})
            ended = ws.receive_json()
            assert ended["type"] == "ended" and ended["summary"]["session_id"] == "sess1"

        # memory written in the background survived, and the session summary was stored
        memories = client.get("/users/u1/memories").json()
        assert [m["key"] for m in memories] == ["language_preference"]


def test_new_session_id_is_generated_when_omitted(container):
    with TestClient(create_app(container)) as client:
        with client.websocket_connect("/ws/chat?user_id=u1") as ws:
            assert len(ws.receive_json()["session_id"]) == 12


def test_bad_json_and_unknown_types_get_an_error_event_without_closing(container):
    with TestClient(create_app(container)) as client:
        with client.websocket_connect("/ws/chat?user_id=u1&session_id=s") as ws:
            ws.receive_json()
            ws.send_text("not json")
            assert ws.receive_json()["type"] == "error"
            ws.send_json({"type": "dance"})
            assert ws.receive_json()["type"] == "error"
            ws.send_json({"type": "message", "text": "   "})
            assert ws.receive_json()["type"] == "error"
            ws.send_json({"type": "message", "text": "still alive?"})
            assert _recv_until(ws, "done")


def test_invalid_user_id_is_rejected(container):
    with TestClient(create_app(container)) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/ws/chat?user_id=bad:id"):
                pass
        assert exc.value.code == 4400


def test_abrupt_disconnect_finalises_the_session_after_the_grace_period(make_container):
    container = make_container(ws_disconnect_grace_seconds=0.05)
    with TestClient(create_app(container)) as client:
        with client.websocket_connect("/ws/chat?user_id=u1&session_id=s9") as ws:
            ws.receive_json()
            ws.send_json({"type": "message", "text": "I live in Pune."})
            _recv_until(ws, "done")
        # socket closed without {"type": "end"}; give the grace task a moment to run
        import time
        for _ in range(100):
            if client.get("/users/u1/memories").json() and container.memory_store._summaries:
                break
            time.sleep(0.02)
        assert ("u1", "s9") in container.memory_store._summaries
