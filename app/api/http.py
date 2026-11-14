"""HTTP routes: POST /chat plus session-end and memory-inspection helpers."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.container import Container
from app.schemas import (
    ChatRequest, ChatResponse, EndSessionResponse, IdStr, MemoryOut, SessionSummaryOut,
)

router = APIRouter()


def _container(request: Request) -> Container:
    return request.app.state.container


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    result = await _container(request).chat.chat(body.user_id, body.session_id, body.message)
    return ChatResponse(
        user_id=body.user_id,
        session_id=body.session_id,
        response=result.response,
        context_used=result.context_used,
        degraded=result.degraded,
        debug=result.debug if body.debug else None,
    )


@router.post("/users/{user_id}/sessions/{session_id}/end", response_model=EndSessionResponse)
async def end_session(user_id: IdStr, session_id: IdStr, request: Request) -> EndSessionResponse:
    summary = await _container(request).chat.end_session(user_id, session_id)
    if summary is None:
        return EndSessionResponse(ended=False)
    return EndSessionResponse(
        ended=True,
        summary=SessionSummaryOut(
            session_id=summary.session_id, summary=summary.summary,
            topics=summary.topics, message_count=summary.message_count,
        ),
    )


@router.get("/users/{user_id}/memories", response_model=list[MemoryOut])
async def list_memories(user_id: IdStr, request: Request) -> list[MemoryOut]:
    memories = await _container(request).memory_store.list(user_id)
    return [
        MemoryOut(
            key=m.key, value=m.value, category=m.category, kind=m.kind, confidence=m.confidence,
            version=m.version, updated_at=m.updated_at.isoformat(), history=m.history,
        )
        for m in memories
    ]


@router.delete("/users/{user_id}/memories/{key}")
async def delete_memory(user_id: IdStr, key: str, request: Request) -> dict:
    if not await _container(request).memory_manager.delete(user_id, key):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": key}
