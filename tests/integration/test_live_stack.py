"""Live-stack tests: real vLLM, Redis, MongoDB, Chroma server and the Granite embedding model.

Run on the machine where `docker compose --profile gpu up -d` is running:

    pytest -m integration -v

Settings come from ./.env (same as the app). Data is isolated: a throw-away MongoDB database and Chroma
collection prefix are used and removed afterwards, and every user id is unique to the run. If a service
is unreachable the tests skip with the reason instead of failing.
"""
from __future__ import annotations

import re
import uuid

import httpx
import pytest

from app.config import Settings
from app.container import build_container
from app.llm.openai_compat import OpenAICompatClient
from app.memory.schema import valid_key
from app.services.extractor import EXTRACTION_SCHEMA, MemoryExtractor
from app.stores.inmemory import InMemoryMemoryStore

pytestmark = pytest.mark.integration

JOB_MSG = "I'm planning to switch jobs next year, and I prefer advice in Hindi."


@pytest.fixture(scope="module")
def live() -> Settings:
    """Real settings; skips the whole module if a required service is not reachable."""
    s = Settings()
    problems = []

    def check(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
            problems.append(f"{name}: {exc.__class__.__name__}: {str(exc)[:80]}")

    if s.llm_backend == "vllm":
        check(f"vLLM {s.vllm_base_url}", lambda: httpx.get(f"{s.vllm_base_url}/v1/models", timeout=3).raise_for_status())
    if s.storage_backend == "services":
        import pymongo
        import redis

        check("Redis", lambda: redis.Redis.from_url(s.redis_url, socket_connect_timeout=2).ping())
        check("MongoDB", lambda: pymongo.MongoClient(s.mongo_url, serverSelectionTimeoutMS=2000).admin.command("ping"))
    if s.vector_backend == "http":
        import chromadb

        check("Chroma", lambda: chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port).heartbeat())
    if problems:
        pytest.skip("live stack not available - " + "; ".join(problems))
    return s


@pytest.fixture
async def stack(live):
    """A real container (isolated Mongo DB + Chroma prefix), cleaned up afterwards."""
    isolated = live.model_copy(update={
        "mongo_db": f"hatchi_it_{uuid.uuid4().hex[:8]}",
        "chroma_collection_prefix": f"it{uuid.uuid4().hex[:8]}",
        "sweeper_enabled": False,
    })
    container = build_container(isolated)
    await container.startup()
    try:
        yield container
    finally:
        await container.chat.drain()
        await container.vectors.drop()
        if isolated.storage_backend == "services":
            import pymongo

            pymongo.MongoClient(isolated.mongo_url).drop_database(isolated.mongo_db)
        await container.shutdown()


def uid() -> str:
    return f"it-{uuid.uuid4().hex[:10]}"


# ---- the model server ----------------------------------------------------------------------------------------
@pytest.fixture
async def vllm(live):
    if live.llm_backend != "vllm":
        pytest.skip("LLM_BACKEND is not vllm")
    client = OpenAICompatClient(live.vllm_base_url, live.vllm_model, api_key=live.vllm_api_key,
                                timeout=live.llm_timeout_seconds)
    yield client
    await client.aclose()


async def test_vllm_plain_reply_and_streaming(vllm):
    reply = await vllm.chat([{"role": "user", "content": "Say hello in one short sentence."}])
    assert reply.strip()

    tokens = [t async for t in vllm.stream([{"role": "user", "content": "Count from one to five."}])]
    assert len(tokens) > 1 and "".join(tokens).strip()


async def test_vllm_returns_schema_valid_json(vllm):
    data = await vllm.chat_json(
        [{"role": "system", "content": "Extract memories. Output JSON."},
         {"role": "user", "content": "I'm planning to switch jobs next year."}],
        EXTRACTION_SCHEMA,
    )
    assert isinstance(data["operations"], list)
    for op in data["operations"]:
        assert op["action"] in ("save", "delete") and isinstance(op["key"], str) and isinstance(op["explicit"], bool)


async def test_llm_extracts_a_memory_the_rules_cannot(vllm):
    """Quality check for the model itself: no regex covers health, so only the LLM can produce this."""
    extractor = MemoryExtractor(vllm, InMemoryMemoryStore())
    ops = await extractor.extract("u", "I've been dealing with chronic back pain and want to get healthier this year.")
    assert ops, "the model returned no memories at all"
    assert all(valid_key(op.key) for op in ops)                       # vocabulary + grounding guards held
    assert any(op.key == "health_focus" for op in ops), f"expected health_focus, got {[(o.key, o.value) for o in ops]}"


# ---- the embedding model ----------------------------------------------------------------------------------------
def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))                           # vectors are L2-normalised


def test_granite_ranks_relevant_memories_higher_in_english_and_hindi(live):
    pytest.importorskip("sentence_transformers")
    from app.embeddings.base import GraniteEmbedder

    embedder = GraniteEmbedder(live.embedding_model, live.embedding_query_prefix, live.embedding_doc_prefix)
    career, hobby = embedder.embed_documents(["career goal: planning to switch jobs next year", "interests: cricket, cooking"])
    assert len(career) == 384

    for question in ("What should I focus on for my career?", "मेरे करियर के बारे में क्या सलाह है?"):
        q = embedder.embed_query(question)
        c, h = _cos(q, career), _cos(q, hobby)
        print(f"{question!r}: career={c:.3f} hobby={h:.3f}")        # use these to tune MEMORY_MIN_SIMILARITY
        assert c > h


# ---- the whole system ---------------------------------------------------------------------------------------------
async def test_memory_saved_in_one_session_is_recalled_in_another(stack):
    user = uid()
    first = await stack.chat.chat(user, "session-A", JOB_MSG)
    assert first.response.strip()
    await stack.chat.drain()

    memories = {m.key: m.value for m in await stack.memory_store.list(user)}
    assert "career_goal" in memories and memories["language_preference"] == "Hindi"
    assert (await stack.sessions.recent(user, "session-A"))[0].content == JOB_MSG      # real Redis window

    second = await stack.chat.chat(user, "session-B", "What do you remember about my career plans?", )
    await stack.chat.drain()
    assert "career_goal" in second.context_used
    assert re.search(r"job|career|switch|नौकरी|करियर|बदल", second.response, re.IGNORECASE), second.response


async def test_irrelevant_memory_is_kept_out_with_real_embeddings(stack):
    user = uid()
    await stack.chat.chat(user, "s1", "I love cricket and cooking.")
    await stack.chat.chat(user, "s1", "I'm planning to switch jobs next year.")
    await stack.chat.drain()

    result = await stack.chat.chat(user, "s2", "What should I focus on for my career?")
    assert "career_goal" in result.context_used
    assert "interests" not in result.context_used, result.debug["memory_scores"]


async def test_correction_replaces_the_old_value_everywhere(stack):
    user = uid()
    await stack.chat.chat(user, "s1", "I prefer advice in Hindi.")
    await stack.chat.drain()
    await stack.chat.chat(user, "s1", "Actually, I prefer advice in English.")
    await stack.chat.drain()

    memory = await stack.memory_store.get(user, "language_preference")
    assert memory.value == "English" and memory.version == 2 and memory.history[0]["value"] == "Hindi"
    hits = await stack.vectors.search_memories(user, "preferred language", k=5)
    assert [h.document for h in hits] == ["preferred language: English"]


async def test_ended_session_is_summarised_and_found_by_a_later_session(stack):
    user = uid()
    await stack.chat.chat(user, "s1", "I'm planning to switch jobs next year and I'm nervous about the timing.")
    await stack.chat.drain()

    summary = await stack.chat.end_session(user, "s1")
    assert summary is not None and summary.summary.strip() and summary.indexed is True
    assert await stack.sessions.recent(user, "s1") == []                                # Redis state cleared
    assert (await stack.memory_store.get_summary(user, "s1")).summary == summary.summary

    result = await stack.chat.chat(user, "s2", "What did we talk about last time regarding my job change?")
    assert "session_summary:s1" in result.context_used, result.debug


async def test_streaming_reply_over_the_real_model(stack):
    tokens: list[str] = []

    async def on_token(token: str) -> None:
        tokens.append(token)

    result = await stack.chat.chat(uid(), "s1", "Give me one sentence of encouragement.", on_token=on_token)
    assert len(tokens) > 1 and "".join(tokens).strip() == result.response and not result.degraded
