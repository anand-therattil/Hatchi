"""MongoDB-backed long-term memory (source of truth) and finished-session summaries.

Collections:
  user_memories      one document per (user_id, key); _id = "user_id:key"
  session_summaries  one document per (user_id, session_id)

Works with PyMongo's AsyncMongoClient (and with Motor-compatible mocks in tests).
"""
from __future__ import annotations

from typing import Optional

from app.models import Memory, SessionSummary


def _memory_doc(memory: Memory) -> dict:
    doc = memory.model_dump(mode="python")
    doc["_id"] = f"{memory.user_id}:{memory.key}"
    return doc


def _to_memory(doc: dict) -> Memory:
    doc = dict(doc)
    doc.pop("_id", None)
    return Memory.model_validate(doc)


def _summary_doc(summary: SessionSummary) -> dict:
    doc = summary.model_dump(mode="python")
    doc["_id"] = f"{summary.user_id}:{summary.session_id}"
    return doc


def _to_summary(doc: dict) -> SessionSummary:
    doc = dict(doc)
    doc.pop("_id", None)
    return SessionSummary.model_validate(doc)


class MongoMemoryStore:
    def __init__(self, db):
        self._memories = db["user_memories"]
        self._summaries = db["session_summaries"]

    async def ensure_indexes(self) -> None:
        await self._memories.create_index([("user_id", 1), ("category", 1)])
        await self._memories.create_index([("indexed", 1)])
        await self._summaries.create_index([("user_id", 1), ("session_id", 1)], unique=True)
        await self._summaries.create_index([("indexed", 1)])

    # ---- memories --------------------------------------------------------
    async def get(self, user_id: str, key: str) -> Optional[Memory]:
        doc = await self._memories.find_one({"_id": f"{user_id}:{key}"})
        return _to_memory(doc) if doc else None

    async def save(self, memory: Memory) -> None:
        doc = _memory_doc(memory)
        await self._memories.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    async def delete(self, user_id: str, key: str) -> bool:
        result = await self._memories.delete_one({"_id": f"{user_id}:{key}"})
        return result.deleted_count > 0

    async def list(self, user_id: str, categories: Optional[list[str]] = None) -> list[Memory]:
        query: dict = {"user_id": user_id}
        if categories:
            query["category"] = {"$in": categories}
        docs = await self._memories.find(query).to_list(None)
        return sorted((_to_memory(d) for d in docs), key=lambda m: m.updated_at, reverse=True)

    async def mark_indexed(self, user_id: str, key: str, indexed: bool) -> None:
        await self._memories.update_one({"_id": f"{user_id}:{key}"}, {"$set": {"indexed": indexed}})

    async def list_unindexed(self) -> list[Memory]:
        docs = await self._memories.find({"indexed": False}).to_list(None)
        return [_to_memory(d) for d in docs]

    # ---- session summaries -------------------------------------------------
    async def save_summary(self, summary: SessionSummary) -> None:
        doc = _summary_doc(summary)
        await self._summaries.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    async def get_summary(self, user_id: str, session_id: str) -> Optional[SessionSummary]:
        doc = await self._summaries.find_one({"_id": f"{user_id}:{session_id}"})
        return _to_summary(doc) if doc else None

    async def mark_summary_indexed(self, user_id: str, session_id: str, indexed: bool) -> None:
        await self._summaries.update_one(
            {"_id": f"{user_id}:{session_id}"}, {"$set": {"indexed": indexed}}
        )

    async def list_unindexed_summaries(self) -> list[SessionSummary]:
        docs = await self._summaries.find({"indexed": False}).to_list(None)
        return [_to_summary(d) for d in docs]
