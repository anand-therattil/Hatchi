"""Domain models shared across the stores and services."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.memory.schema import spec_for


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    ts: float = Field(default_factory=time.time)


class Memory(BaseModel):
    """One remembered fact about a user (source of truth lives in MongoDB)."""

    user_id: str
    key: str
    value: str
    category: str
    kind: Literal["slot", "list", "fact"]
    confidence: float = 0.9
    version: int = 1
    history: list[dict[str, Any]] = Field(default_factory=list)
    source_session: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    indexed: bool = False  # True once the vector index reflects this version

    @property
    def label(self) -> str:
        spec = spec_for(self.key)
        return spec.label if spec else self.key

    @property
    def text(self) -> str:
        """Text that gets embedded and shown to the LLM."""
        return f"{self.label}: {self.value}"


class MemoryOp(BaseModel):
    """An instruction produced by the extractor: save (add/update) or delete a key."""

    action: Literal["save", "delete"]
    key: str
    value: str = ""
    explicit: bool = True  # stated outright by the user (vs. inferred)


class MemoryChange(BaseModel):
    """What actually happened when an op was applied (useful for logs and tests)."""

    action: Literal["created", "updated", "deleted"]
    key: str
    value: str = ""
    previous: Optional[str] = None


class SessionSummary(BaseModel):
    user_id: str
    session_id: str
    summary: str
    topics: list[str] = Field(default_factory=list)
    message_count: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    indexed: bool = False

    @property
    def text(self) -> str:
        topics = f" Topics: {', '.join(self.topics)}." if self.topics else ""
        return f"{self.summary}{topics}"


class RetrievedMemory(BaseModel):
    memory: Memory
    score: float
    similarity: float
    reasons: list[str]


class RetrievedSummary(BaseModel):
    session_id: str
    summary: str
    similarity: float


class RetrievalResult(BaseModel):
    memories: list[RetrievedMemory] = Field(default_factory=list)
    past_sessions: list[RetrievedSummary] = Field(default_factory=list)
    rolling_summary: str = ""
    debug: dict[str, Any] = Field(default_factory=dict)

    @property
    def context_used(self) -> list[str]:
        used = [r.memory.key for r in self.memories]
        used += [f"session_summary:{s.session_id}" for s in self.past_sessions]
        if self.rolling_summary:
            used.append("conversation_summary")
        return used


class ChatResult(BaseModel):
    response: str
    context_used: list[str]
    degraded: bool = False
    debug: dict[str, Any] = Field(default_factory=dict)
