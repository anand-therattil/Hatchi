"""Chroma-backed semantic index over memories and finished-session summaries.

MongoDB stays the source of truth; this is a search index that can be rebuilt from it.
Embeddings are computed here (not by the Chroma server) so the model is under our control.
Chroma's client is synchronous, so every call is pushed to a worker thread.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from app.embeddings.base import Embedder
from app.models import Memory, SessionSummary


@dataclass
class VectorHit:
    key: str  # memory key, or session_id for summaries
    similarity: float  # 1 - cosine distance
    document: str


class VectorStore:
    def __init__(self, client, embedder: Embedder, prefix: str = "hatchi"):
        self._client = client
        cosine = {"hnsw:space": "cosine"}
        self._memories = client.get_or_create_collection(f"{prefix}_memories", metadata=cosine)
        self._summaries = client.get_or_create_collection(f"{prefix}_session_summaries", metadata=cosine)
        self._embedder = embedder

    async def drop(self) -> None:
        """Delete both collections (test teardown / full re-index)."""
        for collection in (self._memories, self._summaries):
            await asyncio.to_thread(self._client.delete_collection, collection.name)

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _hits(result: dict, meta_key: str) -> list[VectorHit]:
        if not result or not result.get("ids") or not result["ids"][0]:
            return []
        hits = []
        for meta, dist, doc in zip(result["metadatas"][0], result["distances"][0], result["documents"][0]):
            hits.append(VectorHit(key=meta[meta_key], similarity=1.0 - float(dist), document=doc))
        return hits

    def _query(self, collection, text: str, where: dict, k: int) -> dict:
        embedding = self._embedder.embed_query(text)
        return collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["metadatas", "distances", "documents"],
        )

    # ---- memories ----------------------------------------------------------
    async def upsert_memory(self, memory: Memory) -> None:
        def _work() -> None:
            self._memories.upsert(
                ids=[f"{memory.user_id}:{memory.key}"],
                documents=[memory.text],
                embeddings=self._embedder.embed_documents([memory.text]),
                metadatas=[{
                    "user_id": memory.user_id,
                    "key": memory.key,
                    "category": memory.category,
                    "kind": memory.kind,
                }],
            )

        await asyncio.to_thread(_work)

    async def delete_memory(self, user_id: str, key: str) -> None:
        await asyncio.to_thread(self._memories.delete, ids=[f"{user_id}:{key}"])

    async def search_memories(self, user_id: str, query: str, k: int = 10) -> list[VectorHit]:
        result = await asyncio.to_thread(self._query, self._memories, query, {"user_id": user_id}, k)
        return self._hits(result, "key")

    async def find_similar_fact(self, user_id: str, text: str) -> Optional[VectorHit]:
        where = {"$and": [{"user_id": user_id}, {"kind": "fact"}]}
        result = await asyncio.to_thread(self._query, self._memories, text, where, 1)
        hits = self._hits(result, "key")
        return hits[0] if hits else None

    # ---- session summaries -------------------------------------------------
    async def upsert_summary(self, summary: SessionSummary) -> None:
        def _work() -> None:
            self._summaries.upsert(
                ids=[f"{summary.user_id}:{summary.session_id}"],
                documents=[summary.text],
                embeddings=self._embedder.embed_documents([summary.text]),
                metadatas=[{"user_id": summary.user_id, "session_id": summary.session_id}],
            )

        await asyncio.to_thread(_work)

    async def search_summaries(
        self, user_id: str, query: str, exclude_session: str, k: int = 3
    ) -> list[VectorHit]:
        where = {"$and": [{"user_id": user_id}, {"session_id": {"$ne": exclude_session}}]}
        result = await asyncio.to_thread(self._query, self._summaries, query, where, k)
        return self._hits(result, "session_id")
