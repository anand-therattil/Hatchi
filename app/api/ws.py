"""WebSocket chat: WS /ws/chat?user_id=...&session_id=...

Client -> server (JSON):
    {"type": "message", "text": "..."}      send a chat message
    {"type": "end"}                          end the session now (final summary is written)
Server -> client (JSON):
    {"type": "session", "session_id": "..."}                    sent right after connecting
    {"type": "token", "text": "..."}                            streamed reply fragments
    {"type": "done", "response": "...", "context_used": [...], "degraded": false}
    {"type": "ended", "summary": {...} | null}                  after an "end" request
    {"type": "error", "detail": "..."}

State lives in Redis, so a client may reconnect with the same session_id and carry on. An abrupt
disconnect finalises the session after WS_DISCONNECT_GRACE_SECONDS unless the client came back.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from app.container import Container
from app.schemas import MAX_MESSAGE_CHARS, IdStr

log = logging.getLogger("hatchi.ws")
router = APIRouter()
_id_adapter = TypeAdapter(IdStr)


@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket, user_id: str = "", session_id: Optional[str] = None) -> None:
    container: Container = ws.app.state.container
    try:
        user_id = _id_adapter.validate_python(user_id)
        session_id = _id_adapter.validate_python(session_id) if session_id else uuid.uuid4().hex[:12]
    except ValidationError:
        await ws.close(code=4400, reason="invalid or missing user_id/session_id")
        return

    await ws.accept()
    container.connections.add(user_id, session_id)
    await ws.send_json({"type": "session", "session_id": session_id})
    client_gone = False
    ended = False

    async def send(payload: dict) -> None:
        nonlocal client_gone
        if client_gone:
            return
        try:
            await ws.send_json(payload)
        except Exception:  # client vanished mid-stream: keep going so the turn is still saved
            client_gone = True

    try:
        while not client_gone:
            try:
                data = await ws.receive_json()
            except ValueError:
                await send({"type": "error", "detail": "messages must be valid JSON"})
                continue

            kind = data.get("type") if isinstance(data, dict) else None
            if kind == "message":
                text = str(data.get("text", "")).strip()
                if not text or len(text) > MAX_MESSAGE_CHARS:
                    await send({"type": "error", "detail": f"text must be 1-{MAX_MESSAGE_CHARS} characters"})
                    continue

                async def on_token(token: str) -> None:
                    await send({"type": "token", "text": token})

                result = await container.chat.chat(user_id, session_id, text, on_token=on_token)
                await send({
                    "type": "done", "response": result.response,
                    "context_used": result.context_used, "degraded": result.degraded,
                })
            elif kind == "end":
                summary = await container.chat.end_session(user_id, session_id)
                ended = True
                await send({
                    "type": "ended",
                    "summary": None if summary is None else {
                        "session_id": summary.session_id, "summary": summary.summary, "topics": summary.topics,
                    },
                })
                await ws.close()
                return
            else:
                await send({"type": "error", "detail": "unknown message type; use 'message' or 'end'"})
    except WebSocketDisconnect:
        pass
    finally:
        container.connections.discard(user_id, session_id)
        if not ended:
            container.chat.spawn_timer(_finalize_after_grace(container, user_id, session_id))


async def _finalize_after_grace(container: Container, user_id: str, session_id: str) -> None:
    await asyncio.sleep(container.settings.ws_disconnect_grace_seconds)
    if container.connections.is_active(user_id, session_id):
        return  # the client reconnected
    try:
        await container.chat.end_session(user_id, session_id)
    except Exception:
        log.exception("finalising %s/%s after disconnect failed", user_id, session_id)
