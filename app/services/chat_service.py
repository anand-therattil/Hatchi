"""Orchestrates one chat turn and the session lifecycle.

Request path (latency-sensitive):
    load recent turns + rolling summary -> retrieve relevant memories -> build prompt -> LLM -> persist turn
Background path (never blocks the reply; serialised per user so writes can't race):
    extract + apply memories -> fold overflowing history into the rolling summary
Session end (explicit /end, WebSocket close, or idle sweep):
    final summary -> MongoDB + vector index -> clear Redis state
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Awaitable, Callable, Coroutine, Optional

from app.config import Settings
from app.llm.base import LLMClient, LLMError
from app.models import ChatResult, Message, SessionSummary
from app.services.extractor import MemoryExtractor
from app.services.memory_manager import MemoryManager
from app.services.prompt import build_messages
from app.services.retriever import Retriever
from app.services.summarizer import Summarizer
from app.stores.base import MemoryStore, SessionStore
from app.stores.vector_store import VectorStore

log = logging.getLogger("hatchi.chat")

TokenCallback = Callable[[str], Awaitable[None]]
FALLBACK_REPLY = (
    "I'm having trouble reaching my reasoning engine right now, but I've noted your message. "
    "Please try again in a moment."
)


class ChatService:
    def __init__(
        self,
        settings: Settings,
        sessions: SessionStore,
        memory_store: MemoryStore,
        vectors: VectorStore,
        llm: LLMClient,
        extractor: MemoryExtractor,
        memory_manager: MemoryManager,
        retriever: Retriever,
        summarizer: Summarizer,
    ):
        self._s = settings
        self._sessions = sessions
        self._store = memory_store
        self._vectors = vectors
        self._llm = llm
        self._extractor = extractor
        self._memories = memory_manager
        self._retriever = retriever
        self._summarizer = summarizer
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._tasks: set[asyncio.Task] = set()   # real work: awaited on drain/shutdown
        self._timers: set[asyncio.Task] = set()  # delayed work (disconnect grace): cancelled on shutdown

    # ---- one chat turn -------------------------------------------------------
    async def chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        on_token: Optional[TokenCallback] = None,
    ) -> ChatResult:
        recent = await self._sessions.recent(user_id, session_id)
        rolling = await self._sessions.get_summary(user_id, session_id)

        retrieval = await self._retriever.retrieve(user_id, session_id, message, recent)
        retrieval.rolling_summary = rolling
        prompt = build_messages(retrieval, rolling, recent, message)

        degraded = False
        try:
            response = await self._generate(prompt, on_token)
        except LLMError as exc:
            log.warning("LLM unavailable, returning fallback: %s", exc)
            degraded = True
            response = exc.args[1] if len(exc.args) > 1 and exc.args[1] else FALLBACK_REPLY
            if on_token and response == FALLBACK_REPLY:
                await on_token(response)

        turn = [Message(role="user", content=message)]
        if not degraded:  # don't pollute history with the canned fallback
            turn.append(Message(role="assistant", content=response))
        await self._sessions.append(user_id, session_id, turn)

        self.spawn(self._post_process(user_id, session_id, message))
        return ChatResult(
            response=response,
            context_used=retrieval.context_used,
            degraded=degraded,
            debug=retrieval.debug,
        )

    async def _generate(self, prompt: list[dict], on_token: Optional[TokenCallback]) -> str:
        if on_token is None:
            return await self._llm.chat(prompt)
        parts: list[str] = []
        try:
            async for token in self._llm.stream(prompt):
                parts.append(token)
                await on_token(token)
        except LLMError as exc:
            if not parts:
                raise
            # keep the partial answer the user already saw
            raise LLMError("stream interrupted", "".join(parts).strip()) from exc
        text = "".join(parts).strip()
        if not text:
            raise LLMError("empty response")
        return text

    # ---- background work -------------------------------------------------------
    def spawn(self, coro: Coroutine) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def spawn_timer(self, coro: Coroutine) -> asyncio.Task:
        """Like spawn(), for delayed jobs that shutdown should cancel rather than wait for."""
        task = asyncio.create_task(coro)
        self._timers.add(task)
        task.add_done_callback(self._timers.discard)
        return task

    async def drain(self) -> None:
        """Wait for all pending background work (used by tests and graceful shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def cancel_timers(self) -> None:
        """Drop pending grace timers; their sessions are picked up later by the idle sweeper."""
        timers = list(self._timers)
        for task in timers:
            task.cancel()
        await asyncio.gather(*timers, return_exceptions=True)

    async def _post_process(self, user_id: str, session_id: str, message: str) -> None:
        try:
            async with self._locks[user_id]:
                ops = await self._extractor.extract(user_id, message)
                if ops:
                    await self._memories.apply(user_id, session_id, ops)
                await self._maybe_fold(user_id, session_id)
        except Exception:
            log.exception("background post-processing failed for %s/%s", user_id, session_id)

    async def _maybe_fold(self, user_id: str, session_id: str) -> None:
        """Keep the verbatim window bounded: summarise the oldest turns into the rolling summary."""
        recent = await self._sessions.recent(user_id, session_id)
        if len(recent) <= self._s.history_max_messages:
            return
        old = recent[: self._s.history_fold_count]
        previous = await self._sessions.get_summary(user_id, session_id)
        summary = await self._summarizer.fold(previous, old)
        await self._sessions.set_summary(user_id, session_id, summary)
        await self._sessions.drop_oldest(user_id, session_id, len(old))  # only after the summary is stored

    # ---- session lifecycle -----------------------------------------------------
    async def end_session(self, user_id: str, session_id: str) -> Optional[SessionSummary]:
        """Summarise a finished session into long-term storage. Returns None for an empty session."""
        async with self._locks[user_id]:  # queues behind any pending memory extraction for this user
            recent = await self._sessions.recent(user_id, session_id)
            rolling = await self._sessions.get_summary(user_id, session_id)
            if not recent and not rolling:
                await self._sessions.delete(user_id, session_id)
                return None

            meta = await self._sessions.get_meta(user_id, session_id)
            earlier = await self._store.get_summary(user_id, session_id)  # same id reused later
            previous = "\n".join(p for p in (earlier.summary if earlier else "", rolling) if p)

            text, topics = await self._summarizer.finalize(previous, recent)
            summary = SessionSummary(
                user_id=user_id,
                session_id=session_id,
                summary=text,
                topics=topics,
                message_count=meta.get("message_count", 0) + (earlier.message_count if earlier else 0),
            )
            await self._store.save_summary(summary)
            try:
                await self._vectors.upsert_summary(summary)
                await self._store.mark_summary_indexed(user_id, session_id, True)
                summary.indexed = True
            except Exception:
                log.exception("summary vector write failed for %s/%s (retried on startup)", user_id, session_id)
            await self._sessions.delete(user_id, session_id)
            log.info("session ended user=%s session=%s", user_id, session_id)
            return summary

    async def sweep_idle(self, now: Optional[float] = None) -> int:
        """Finalise sessions idle past the threshold (covers clients that never say goodbye)."""
        cutoff = (now or time.time()) - self._s.session_idle_seconds
        ended = 0
        for user_id, session_id in await self._sessions.idle_sessions(cutoff):
            try:
                await self.end_session(user_id, session_id)
                ended += 1
            except Exception:
                log.exception("sweep failed for %s/%s", user_id, session_id)
        return ended
