"""Shared fixtures: the whole service graph wired with in-memory stores, an ephemeral Chroma,
a fake (hashed bag-of-words) embedder and a scriptable mock LLM - no Docker, model or network."""
from __future__ import annotations

import uuid

import chromadb
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.container import Container, assemble
from app.embeddings.base import FakeEmbedder
from app.llm.mock import MockLLM
from app.main import create_app
from app.stores.inmemory import InMemoryMemoryStore, InMemorySessionStore


def make_settings(**overrides) -> Settings:
    base = dict(
        storage_backend="memory", vector_backend="ephemeral", llm_backend="mock", embedding_backend="fake",
        sweeper_enabled=False,
        # the fake embedder only scores word overlap (0..1), so use gentler thresholds than the Granite defaults
        memory_min_similarity=0.45, summary_min_similarity=0.2, summary_recall_min_similarity=0.1,
        chroma_collection_prefix=f"t{uuid.uuid4().hex[:8]}",  # Ephemeral Chroma is process-global
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
def make_container():
    def _make(llm: MockLLM | None = None, **settings_overrides) -> Container:
        return assemble(
            make_settings(**settings_overrides),
            sessions=InMemorySessionStore(),
            memory_store=InMemoryMemoryStore(),
            vector_client=chromadb.EphemeralClient(),
            embedder=FakeEmbedder(),
            llm=llm or MockLLM(),
        )
    return _make


@pytest.fixture
def container(make_container) -> Container:
    return make_container()


@pytest.fixture
async def client(container):
    app = create_app(container)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def say(client: AsyncClient, container: Container, user: str, session: str, text: str, **extra) -> dict:
    """POST /chat, then wait for the background memory/summary work to finish."""
    resp = await client.post("/chat", json={"user_id": user, "session_id": session, "message": text, **extra})
    assert resp.status_code == 200, resp.text
    await container.chat.drain()
    return resp.json()
