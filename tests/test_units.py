"""Focused unit tests: extraction rules/guards, vector store, and the real Redis/Mongo store code paths."""
from __future__ import annotations

import chromadb
import pytest

from app.embeddings.base import FakeEmbedder
from app.llm.mock import MockLLM
from app.memory.categories import detect_categories, has_recall_intent, is_followup
from app.memory.rules import candidate_statements, rule_extract
from app.models import Memory, Message, SessionSummary
from app.services.extractor import MemoryExtractor
from app.stores.inmemory import InMemoryMemoryStore
from app.stores.vector_store import VectorStore


# ---- gate + rules ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "What should I focus on for my career?",
    "Why do you say that?",
    "Can you tell me in Hindi?",
    "Tell me about Saturn",
])
def test_questions_and_non_personal_text_never_pass_the_gate(text):
    assert candidate_statements(text) == []


def test_gate_keeps_only_the_declarative_sentence():
    text = "I love cricket. What should I do about my career?"
    assert candidate_statements(text) == ["I love cricket."]


@pytest.mark.parametrize("text,key,expected", [
    ("I prefer advice in Hindi.", "language_preference", "Hindi"),
    ("Please reply in English from now on", "language_preference", "English"),
    ("I'm planning to switch jobs next year", "career_goal", "planning to switch jobs next year"),
    ("I work as a software engineer at Infosys.", "current_job", "work as a software engineer at Infosys"),
    ("I live in Pune.", "location", "Pune"),
    ("I'm married.", "relationship_status", "married"),
    ("I love cricket and cooking.", "interests", "cricket, cooking"),
    ("I was born on 5 May 1994 in Pune.", "birth_details", "born on 5 May 1994 in Pune"),
])
def test_rules_extract_obvious_facts(text, key, expected):
    ops = {o.key: o for o in rule_extract(candidate_statements(text))}
    assert ops[key].value == expected and ops[key].action == "save"


def test_rules_split_a_compound_sentence_into_two_memories():
    ops = {o.key: o.value for o in rule_extract(candidate_statements(
        "I'm planning to switch jobs next year, and I prefer advice in Hindi."))}
    assert ops == {"career_goal": "planning to switch jobs next year", "language_preference": "Hindi"}


def test_negated_language_is_not_a_preference():
    assert rule_extract(candidate_statements("I do not want advice in English.")) == []


# ---- LLM extraction guards ----------------------------------------------------------------------
async def test_llm_output_is_validated_and_grounded():
    hallucinated = {"operations": [
        {"action": "save", "key": "career_goal", "value": "become an astronaut", "explicit": True},   # not in text
        {"action": "save", "key": "made_up_key", "value": "switching jobs", "explicit": True},        # not in vocabulary
        {"action": "save", "key": "health_focus", "value": "knee pain recovery", "explicit": True},    # grounded
        {"action": "delete", "key": "location", "value": "", "explicit": True},                        # nothing to delete
    ]}
    extractor = MemoryExtractor(MockLLM(json_handler=lambda m, s: hallucinated), InMemoryMemoryStore())
    ops = await extractor.extract("u", "I'm worried about my knee pain recovery.")
    assert [(o.key, o.value) for o in ops] == [("health_focus", "knee pain recovery")]


async def test_rules_win_over_llm_for_the_same_slot():
    llm_says = {"operations": [{"action": "save", "key": "language_preference", "value": "Hindi maybe", "explicit": False}]}
    extractor = MemoryExtractor(MockLLM(json_handler=lambda m, s: llm_says), InMemoryMemoryStore())
    ops = await extractor.extract("u", "I prefer advice in Hindi.")
    assert [(o.key, o.value, o.explicit) for o in ops] == [("language_preference", "Hindi", True)]


async def test_extractor_survives_llm_failure():
    llm = MockLLM()
    llm.fail = True
    ops = await MemoryExtractor(llm, InMemoryMemoryStore()).extract("u", "I prefer advice in Hindi.")
    assert [o.key for o in ops] == ["language_preference"]


# ---- query understanding ----------------------------------------------------------------------------
def test_category_detection_and_intents():
    assert detect_categories("What should I focus on for my career?") == ["career"]
    assert detect_categories("meri shaadi kab hogi") == ["relationships"]
    assert detect_categories("tell me about Saturn") == []
    assert has_recall_intent("What do you remember about me?")
    assert is_followup("Why do you say that?") and not is_followup("What should I focus on for my career?")


# ---- vector store -------------------------------------------------------------------------------------
async def test_vector_store_is_filtered_per_user_and_ranks_by_similarity():
    store = VectorStore(chromadb.EphemeralClient(), FakeEmbedder(), prefix="vec_unit")
    mk = lambda user, key, value, cat: Memory(user_id=user, key=key, value=value, category=cat, kind="slot")
    await store.upsert_memory(mk("a", "career_goal", "planning to switch jobs", "career"))
    await store.upsert_memory(mk("a", "location", "Pune", "personal"))
    await store.upsert_memory(mk("b", "career_goal", "switch jobs too", "career"))

    hits = await store.search_memories("a", "switch jobs career", k=5)
    assert [h.key for h in hits][0] == "career_goal"
    assert len(hits) == 2                                  # user b's memory never leaks into a's results
    assert hits[0].similarity > hits[1].similarity

    await store.delete_memory("a", "career_goal")
    assert "career_goal" not in [h.key for h in await store.search_memories("a", "switch jobs", k=5)]


async def test_summary_search_excludes_the_current_session():
    store = VectorStore(chromadb.EphemeralClient(), FakeEmbedder(), prefix="vec_unit2")
    for sid in ("s1", "s2"):
        await store.upsert_summary(SessionSummary(user_id="a", session_id=sid, summary="career switch plans"))
    hits = await store.search_summaries("a", "career switch", exclude_session="s1", k=5)
    assert [h.key for h in hits] == ["s2"]


# ---- real store implementations against fakes -------------------------------------------------------------
async def test_redis_session_store_window_summary_and_idle_tracking():
    fakeredis = pytest.importorskip("fakeredis")
    from app.stores.redis_session_store import RedisSessionStore

    store = RedisSessionStore(fakeredis.FakeAsyncRedis(decode_responses=True), ttl_seconds=60)
    n = await store.append("u", "s", [Message(role="user", content="one"), Message(role="assistant", content="two")])
    assert n == 2
    await store.append("u", "s", [Message(role="user", content="three")])
    assert [m.content for m in await store.recent("u", "s")] == ["one", "two", "three"]

    await store.drop_oldest("u", "s", 2)
    assert [m.content for m in await store.recent("u", "s")] == ["three"]

    await store.set_summary("u", "s", "earlier stuff")
    assert await store.get_summary("u", "s") == "earlier stuff"
    assert (await store.get_meta("u", "s"))["message_count"] == 3
    assert await store.idle_sessions(last_active_before=1) == []
    assert await store.idle_sessions(last_active_before=9_999_999_999) == [("u", "s")]

    await store.delete("u", "s")
    assert await store.recent("u", "s") == [] and await store.idle_sessions(9_999_999_999) == []


async def test_mongo_memory_store_roundtrip_and_unindexed_tracking():
    mongomock = pytest.importorskip("mongomock_motor")
    from app.stores.mongo_memory_store import MongoMemoryStore

    store = MongoMemoryStore(mongomock.AsyncMongoMockClient()["hatchi_test"])
    await store.ensure_indexes()
    memory = Memory(user_id="u", key="career_goal", value="switch jobs", category="career", kind="slot")
    await store.save(memory)

    got = await store.get("u", "career_goal")
    assert got.value == "switch jobs" and got.indexed is False
    assert [m.key for m in await store.list_unindexed()] == ["career_goal"]
    await store.mark_indexed("u", "career_goal", True)
    assert await store.list_unindexed() == []

    await store.save(got.model_copy(update={"value": "start a business", "version": 2}))   # upsert on same key
    assert [m.value for m in await store.list("u")] == ["start a business"]
    assert await store.list("u", categories=["finance"]) == []
    assert await store.delete("u", "career_goal") and not await store.delete("u", "career_goal")

    await store.save_summary(SessionSummary(user_id="u", session_id="s", summary="talked careers"))
    assert (await store.get_summary("u", "s")).summary == "talked careers"
