"""HTTP/WebSocket request and response schemas."""
from __future__ import annotations

from typing import Annotated, Any, Optional

from pydantic import BaseModel, StringConstraints, field_validator

# ids end up in Redis/Mongo keys, so keep them to a safe charset (no ':' etc.)
IdStr = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.@\-]{1,128}$")]
ID_PATTERN = r"^[A-Za-z0-9_.@\-]{1,128}$"

MAX_MESSAGE_CHARS = 4000


class ChatRequest(BaseModel):
    user_id: IdStr
    session_id: IdStr
    message: str
    debug: bool = False  # include retrieval diagnostics in the response

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        if len(value) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message must be at most {MAX_MESSAGE_CHARS} characters")
        return value


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    response: str
    context_used: list[str]
    degraded: bool = False
    debug: Optional[dict[str, Any]] = None


class SessionSummaryOut(BaseModel):
    session_id: str
    summary: str
    topics: list[str]
    message_count: int


class EndSessionResponse(BaseModel):
    ended: bool
    summary: Optional[SessionSummaryOut] = None


class MemoryOut(BaseModel):
    key: str
    value: str
    category: str
    kind: str
    confidence: float
    version: int
    updated_at: str
    history: list[dict[str, Any]]
