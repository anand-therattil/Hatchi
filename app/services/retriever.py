"""Picks the context worth giving the LLM for the current message.

Sources, each with its own mechanism:
  * current session  -> not retrieval: recent turns + rolling summary go straight into the prompt
  * long-term facts  -> category match (keywords) UNION semantic search (Chroma), then scored + capped
  * past sessions    -> semantic search over finished-session summaries, excluding the live session

Selection rules for a memory (any one qualifies):
  always-on slot (language preference) | category matches the question
  | semantic similarity >= threshold   | "what do you remember" with no category -> everything (capped)
Anything else is dropped, which is how an unrelated interest stays out of a career answer.
"""
from __future__ import annotations

from typing import Optional

from app.config import Settings
from app.memory.categories import detect_categories, has_recall_intent, is_followup
from app.memory.schema import ALWAYS_INCLUDE_KEYS
from app.models import Memory, Message, RetrievalResult, RetrievedMemory, RetrievedSummary
from app.stores.base import MemoryStore
from app.stores.vector_store import VectorStore


class Retriever:
    def __init__(self, settings: Settings, store: MemoryStore, vectors: VectorStore):
        self._s = settings
        self._store = store
        self._vectors = vectors

    async def retrieve(
        self, user_id: str, session_id: str, message: str, recent: list[Message]
    ) -> RetrievalResult:
        s = self._s
        prev_user = next((m.content for m in reversed(recent) if m.role == "user"), None)
        followup = is_followup(message) and prev_user is not None
        categories = detect_categories(message)
        if not categories and followup:
            categories = detect_categories(prev_user)  # "Why do you say that?" inherits the topic
        recall = has_recall_intent(message)
        query = f"{prev_user}\n{message}" if followup else message

        # ---- candidate memories -------------------------------------------------
        candidates: dict[str, tuple[Memory, float]] = {}

        async def add(memory: Optional[Memory], similarity: float = 0.0) -> None:
            if memory is None:
                return
            prior = candidates.get(memory.key)
            if prior is None or similarity > prior[1]:
                candidates[memory.key] = (memory, similarity)

        if categories:
            for memory in await self._store.list(user_id, categories):
                await add(memory)
        if recall and not categories:
            for memory in await self._store.list(user_id):
                await add(memory)
        for key in ALWAYS_INCLUDE_KEYS:
            await add(await self._store.get(user_id, key))
        for hit in await self._vectors.search_memories(user_id, query, k=10):
            memory = candidates[hit.key][0] if hit.key in candidates else await self._store.get(user_id, hit.key)
            await add(memory, hit.similarity)  # a vector hit with no Mongo doc is stale: skipped

        # ---- score + select ------------------------------------------------------
        always, others = [], []
        for memory, sim in candidates.values():
            reasons = []
            if memory.key in ALWAYS_INCLUDE_KEYS:
                reasons.append("always_on")
            if memory.category in categories:
                reasons.append("category")
            if sim >= s.memory_min_similarity:
                reasons.append("semantic")
            if recall and not categories:
                reasons.append("recall")
            if not reasons:
                continue
            score = sim + (s.category_boost if "category" in reasons else 0.0) + 0.05 * memory.confidence
            item = RetrievedMemory(memory=memory, score=round(score, 4), similarity=round(sim, 4), reasons=reasons)
            (always if "always_on" in reasons else others).append(item)

        cap = s.memory_recall_max if (recall and not categories) else s.memory_top_k
        others.sort(key=lambda r: r.score, reverse=True)
        selected = always + others[:cap]

        # ---- past session summaries -----------------------------------------------
        threshold = s.summary_recall_min_similarity if recall else s.summary_min_similarity
        past = [
            RetrievedSummary(session_id=h.key, summary=h.document, similarity=round(h.similarity, 4))
            for h in await self._vectors.search_summaries(user_id, query, session_id, k=s.summary_top_k)
            if h.similarity >= threshold
        ]

        return RetrievalResult(
            memories=selected,
            past_sessions=past,
            debug={
                "categories": categories,
                "followup": followup,
                "recall_intent": recall,
                "query": query,
                "considered": len(candidates),
                "excluded": sorted(k for k in candidates if k not in {r.memory.key for r in selected}),
                "memory_scores": {r.memory.key: {"score": r.score, "sim": r.similarity, "why": r.reasons} for r in selected},
            },
        )
