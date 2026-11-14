"""In-process implementations of the store interfaces.

Used for tests and for running the service with no infrastructure (STORAGE_BACKEND=memory).
"""
from __future__ import annotations

import time
from typing import Optional

from app.models import Memory, Message, SessionSummary


class InMemorySessionStore:
    def __init__(self) -> None:
        self._messages: dict[tuple[str, str], list[Message]] = {}
        self._summary: dict[tuple[str, str], str] = {}
        self._meta: dict[tuple[str, str], dict] = {}

    async def append(self, user_id: str, session_id: str, messages: list[Message]) -> int:
        k = (user_id, session_id)
        now = time.time()
        self._messages.setdefault(k, []).extend(m.model_copy() for m in messages)
        meta = self._meta.setdefault(k, {"started_at": now, "last_active": now, "message_count": 0})
        meta["last_active"] = now
        meta["message_count"] += len(messages)
        return len(self._messages[k])

    async def recent(self, user_id: str, session_id: str) -> list[Message]:
        return [m.model_copy() for m in self._messages.get((user_id, session_id), [])]

    async def drop_oldest(self, user_id: str, session_id: str, count: int) -> None:
        k = (user_id, session_id)
        if count > 0 and k in self._messages:
            self._messages[k] = self._messages[k][count:]

    async def get_summary(self, user_id: str, session_id: str) -> str:
        return self._summary.get((user_id, session_id), "")

    async def set_summary(self, user_id: str, session_id: str, summary: str) -> None:
        self._summary[(user_id, session_id)] = summary

    async def get_meta(self, user_id: str, session_id: str) -> dict:
        return dict(self._meta.get((user_id, session_id), {"started_at": 0.0, "last_active": 0.0, "message_count": 0}))

    async def idle_sessions(self, last_active_before: float) -> list[tuple[str, str]]:
        return [k for k, meta in self._meta.items() if meta["last_active"] <= last_active_before]

    async def delete(self, user_id: str, session_id: str) -> None:
        k = (user_id, session_id)
        self._messages.pop(k, None)
        self._summary.pop(k, None)
        self._meta.pop(k, None)

    # test helper
    def force_last_active(self, user_id: str, session_id: str, ts: float) -> None:
        self._meta[(user_id, session_id)]["last_active"] = ts


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._memories: dict[tuple[str, str], Memory] = {}
        self._summaries: dict[tuple[str, str], SessionSummary] = {}

    async def ensure_indexes(self) -> None:
        return None

    async def get(self, user_id: str, key: str) -> Optional[Memory]:
        m = self._memories.get((user_id, key))
        return m.model_copy(deep=True) if m else None

    async def save(self, memory: Memory) -> None:
        self._memories[(memory.user_id, memory.key)] = memory.model_copy(deep=True)

    async def delete(self, user_id: str, key: str) -> bool:
        return self._memories.pop((user_id, key), None) is not None

    async def list(self, user_id: str, categories: Optional[list[str]] = None) -> list[Memory]:
        items = [m for (u, _), m in self._memories.items() if u == user_id]
        if categories:
            items = [m for m in items if m.category in categories]
        return sorted((m.model_copy(deep=True) for m in items), key=lambda m: m.updated_at, reverse=True)

    async def mark_indexed(self, user_id: str, key: str, indexed: bool) -> None:
        if (user_id, key) in self._memories:
            self._memories[(user_id, key)].indexed = indexed

    async def list_unindexed(self) -> list[Memory]:
        return [m.model_copy(deep=True) for m in self._memories.values() if not m.indexed]

    async def save_summary(self, summary: SessionSummary) -> None:
        self._summaries[(summary.user_id, summary.session_id)] = summary.model_copy(deep=True)

    async def get_summary(self, user_id: str, session_id: str) -> Optional[SessionSummary]:
        s = self._summaries.get((user_id, session_id))
        return s.model_copy(deep=True) if s else None

    async def mark_summary_indexed(self, user_id: str, session_id: str, indexed: bool) -> None:
        if (user_id, session_id) in self._summaries:
            self._summaries[(user_id, session_id)].indexed = indexed

    async def list_unindexed_summaries(self) -> list[SessionSummary]:
        return [s.model_copy(deep=True) for s in self._summaries.values() if not s.indexed]
