"""Applies memory operations: upserts, corrections, deletes, and keeps the vector index in sync.

Update semantics
  * slot  - a new value replaces the old one; the old value is archived in `history`, `version` is bumped.
  * list  - values are merged (case-insensitive de-dupe); no-op if nothing new.
  * fact  - near-duplicate free-text notes (cosine >= FACT_DEDUPE_SIMILARITY) update the existing note.
  * An *inferred* value (confidence 0.6) never overwrites an *explicit* one (0.9); an explicit
    statement always wins, which is how user corrections ("actually, I prefer English") land.

MongoDB is written first (source of truth). If the vector write fails the memory is flagged
`indexed=False` and `reindex_pending()` retries it on the next startup.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from app.config import Settings
from app.memory.schema import NEW_FACT_KEY, spec_for
from app.models import Memory, MemoryChange, MemoryOp, utcnow
from app.stores.base import MemoryStore
from app.stores.vector_store import VectorStore

log = logging.getLogger("hatchi.memory")

EXPLICIT_CONFIDENCE = 0.9
INFERRED_CONFIDENCE = 0.6
HISTORY_LIMIT = 10


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _merge_list(existing: str, new: str) -> str:
    items = [i.strip() for i in existing.split(",") if i.strip()]
    seen = {_norm(i) for i in items}
    for item in (i.strip() for i in new.split(",")):
        if item and _norm(item) not in seen:
            items.append(item)
            seen.add(_norm(item))
    return ", ".join(items)


class MemoryManager:
    def __init__(self, settings: Settings, store: MemoryStore, vectors: VectorStore):
        self._settings = settings
        self._store = store
        self._vectors = vectors

    # ---- public API --------------------------------------------------------
    async def apply(self, user_id: str, session_id: Optional[str], ops: list[MemoryOp]) -> list[MemoryChange]:
        changes: list[MemoryChange] = []
        for op in ops:
            try:
                change = await self._apply_one(user_id, session_id, op)
            except Exception:  # one bad op must not lose the rest
                log.exception("failed to apply memory op %s", op)
                continue
            if change:
                log.info("memory %s user=%s key=%s", change.action, user_id, change.key)
                changes.append(change)
        return changes

    async def delete(self, user_id: str, key: str) -> bool:
        existing = await self._store.get(user_id, key)
        if not existing:
            return False
        await self._store.delete(user_id, key)
        await self._vectors.delete_memory(user_id, key)
        return True

    async def reindex_pending(self) -> int:
        """Retry vector writes that failed earlier. Returns how many items were re-indexed."""
        count = 0
        for memory in await self._store.list_unindexed():
            if await self._index_memory(memory):
                count += 1
        for summary in await self._store.list_unindexed_summaries():
            try:
                await self._vectors.upsert_summary(summary)
                await self._store.mark_summary_indexed(summary.user_id, summary.session_id, True)
                count += 1
            except Exception:
                log.exception("reindex of summary %s failed", summary.session_id)
        return count

    # ---- internals ---------------------------------------------------------
    async def _apply_one(self, user_id: str, session_id: Optional[str], op: MemoryOp) -> Optional[MemoryChange]:
        spec = spec_for(op.key)
        if spec is None:
            return None

        if op.action == "delete":
            existing = await self._store.get(user_id, op.key)
            if not existing:
                return None
            await self.delete(user_id, op.key)
            return MemoryChange(action="deleted", key=op.key, previous=existing.value)

        value = re.sub(r"\s+", " ", op.value).strip()
        if not value:
            return None
        confidence = EXPLICIT_CONFIDENCE if op.explicit else INFERRED_CONFIDENCE

        key = op.key
        if spec.kind == "fact" and key == NEW_FACT_KEY:
            key = await self._resolve_fact_key(user_id, value)

        existing = await self._store.get(user_id, key)
        if existing is None:
            memory = Memory(
                user_id=user_id, key=key, value=value, category=spec.category, kind=spec.kind,
                confidence=confidence, source_session=session_id,
            )
            await self._persist(memory)
            return MemoryChange(action="created", key=key, value=value)

        if spec.kind == "list":
            merged = _merge_list(existing.value, value)
            if _norm(merged) == _norm(existing.value):
                return None
            new_value, new_conf = merged, max(existing.confidence, confidence)
        else:
            if _norm(value) == _norm(existing.value):
                return None
            if confidence < existing.confidence:  # inferred must not clobber explicit
                return None
            new_value, new_conf = value, confidence

        history = (existing.history + [{
            "value": existing.value,
            "version": existing.version,
            "confidence": existing.confidence,
            "replaced_at": utcnow().isoformat(),
        }])[-HISTORY_LIMIT:]
        updated = existing.model_copy(update={
            "value": new_value,
            "confidence": new_conf,
            "version": existing.version + 1,
            "history": history,
            "source_session": session_id,
            "updated_at": utcnow(),
            "indexed": False,
        })
        await self._persist(updated)
        return MemoryChange(action="updated", key=key, value=new_value, previous=existing.value)

    async def _resolve_fact_key(self, user_id: str, value: str) -> str:
        """Reuse the key of a near-duplicate note, otherwise mint a stable new one."""
        try:
            hit = await self._vectors.find_similar_fact(user_id, f"note: {value}")
        except Exception:
            log.exception("fact de-dupe lookup failed")
            hit = None
        if hit and hit.similarity >= self._settings.fact_dedupe_similarity:
            return hit.key
        return "fact_" + hashlib.sha1(_norm(value).encode()).hexdigest()[:8]

    async def _persist(self, memory: Memory) -> None:
        memory.indexed = False
        await self._store.save(memory)
        await self._index_memory(memory)

    async def _index_memory(self, memory: Memory) -> bool:
        try:
            await self._vectors.upsert_memory(memory)
            await self._store.mark_indexed(memory.user_id, memory.key, True)
            return True
        except Exception:
            log.exception("vector index write failed for %s:%s (will retry on startup)", memory.user_id, memory.key)
            return False
