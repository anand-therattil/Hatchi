"""Wires settings -> concrete stores/providers -> services."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config import Settings
from app.embeddings.base import Embedder, FakeEmbedder, GraniteEmbedder
from app.llm.base import LLMClient
from app.llm.mock import MockLLM
from app.llm.ollama import OllamaClient
from app.llm.openai_compat import OpenAICompatClient
from app.services.chat_service import ChatService
from app.services.extractor import MemoryExtractor
from app.services.memory_manager import MemoryManager
from app.services.retriever import Retriever
from app.services.summarizer import Summarizer
from app.stores.base import MemoryStore, SessionStore
from app.stores.inmemory import InMemoryMemoryStore, InMemorySessionStore
from app.stores.vector_store import VectorStore

log = logging.getLogger("hatchi.container")


class ConnectionRegistry:
    """Tracks live WebSocket connections so a brief disconnect doesn't end the session."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}

    def add(self, user_id: str, session_id: str) -> None:
        k = (user_id, session_id)
        self._counts[k] = self._counts.get(k, 0) + 1

    def discard(self, user_id: str, session_id: str) -> None:
        k = (user_id, session_id)
        if k in self._counts:
            self._counts[k] -= 1
            if self._counts[k] <= 0:
                del self._counts[k]

    def is_active(self, user_id: str, session_id: str) -> bool:
        return (user_id, session_id) in self._counts


@dataclass
class Container:
    settings: Settings
    sessions: SessionStore
    memory_store: MemoryStore
    vectors: VectorStore
    llm: LLMClient
    memory_manager: MemoryManager
    chat: ChatService
    connections: ConnectionRegistry = field(default_factory=ConnectionRegistry)
    _closers: list = field(default_factory=list)
    _sweeper: Optional[asyncio.Task] = None

    async def startup(self) -> None:
        await self.memory_store.ensure_indexes()
        retried = await self.memory_manager.reindex_pending()
        if retried:
            log.info("re-indexed %d pending item(s)", retried)
        if self.settings.sweeper_enabled:
            self._sweeper = asyncio.create_task(self._sweep_loop())

    async def shutdown(self) -> None:
        if self._sweeper:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)
        await self.chat.cancel_timers()
        await self.chat.drain()
        for close in self._closers:
            try:
                await close()
            except Exception:
                log.exception("error while closing a client")

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.sweep_interval_seconds)
            try:
                await self.chat.sweep_idle()
            except Exception:
                log.exception("idle sweep failed")


def assemble(
    settings: Settings,
    *,
    sessions: SessionStore,
    memory_store: MemoryStore,
    vector_client,
    embedder: Embedder,
    llm: LLMClient,
    closers: Optional[list] = None,
) -> Container:
    """Build the service graph from already-constructed infrastructure (also used by tests)."""
    vectors = VectorStore(vector_client, embedder, prefix=settings.chroma_collection_prefix)
    manager = MemoryManager(settings, memory_store, vectors)
    chat = ChatService(
        settings=settings,
        sessions=sessions,
        memory_store=memory_store,
        vectors=vectors,
        llm=llm,
        extractor=MemoryExtractor(llm, memory_store),
        memory_manager=manager,
        retriever=Retriever(settings, memory_store, vectors),
        summarizer=Summarizer(llm),
    )
    return Container(
        settings=settings, sessions=sessions, memory_store=memory_store, vectors=vectors,
        llm=llm, memory_manager=manager, chat=chat, _closers=closers or [],
    )


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "fake":
        return FakeEmbedder()
    return GraniteEmbedder(settings.embedding_model, settings.embedding_query_prefix, settings.embedding_doc_prefix)


def build_container(settings: Settings) -> Container:
    closers: list = []

    # ---- session + long-term stores -------------------------------------------
    if settings.storage_backend == "memory":
        sessions: SessionStore = InMemorySessionStore()
        memory_store: MemoryStore = InMemoryMemoryStore()
    else:
        import redis.asyncio as aioredis
        from pymongo import AsyncMongoClient

        from app.stores.mongo_memory_store import MongoMemoryStore
        from app.stores.redis_session_store import RedisSessionStore

        redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
        mongo_client = AsyncMongoClient(settings.mongo_url, tz_aware=True)
        closers += [redis_client.aclose, mongo_client.close]
        sessions = RedisSessionStore(redis_client, ttl_seconds=settings.session_ttl_seconds)
        memory_store = MongoMemoryStore(mongo_client[settings.mongo_db])

    # ---- vector index ---------------------------------------------------------
    import chromadb

    if settings.vector_backend == "http":
        vector_client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    elif settings.vector_backend == "persistent":
        vector_client = chromadb.PersistentClient(path=settings.chroma_path)
    else:
        vector_client = chromadb.EphemeralClient()

    # ---- providers ------------------------------------------------------------
    embedder = build_embedder(settings)

    if settings.llm_backend == "mock":
        llm: LLMClient = MockLLM()
    elif settings.llm_backend == "ollama":
        ollama = OllamaClient(
            settings.ollama_base_url,
            settings.ollama_model,
            timeout=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
        )
        closers.append(ollama.aclose)
        llm = ollama
    else:  # vllm
        vllm = OpenAICompatClient(
            settings.vllm_base_url,
            settings.vllm_model,
            api_key=settings.vllm_api_key,
            timeout=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        closers.append(vllm.aclose)
        llm = vllm

    return assemble(
        settings, sessions=sessions, memory_store=memory_store, vector_client=vector_client,
        embedder=embedder, llm=llm, closers=closers,
    )
