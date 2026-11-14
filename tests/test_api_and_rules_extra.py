"""Remaining API endpoints, more extraction-rule edge cases, query keywords and the Ollama client."""
from __future__ import annotations

import json

import httpx
import pytest

from app.llm.base import LLMError
from app.llm.ollama import OllamaClient
from app.memory.categories import detect_categories, has_recall_intent, is_followup
from app.memory.rules import candidate_statements, rule_extract
from tests.conftest import say


# ---- HTTP endpoints -----------------------------------------------------------------------------------------
async def test_health(client):
    assert (await client.get("/health")).json() == {"status": "ok"}


async def test_debug_block_is_only_returned_when_requested(client, container):
    plain = await say(client, container, "u", "s", "hello")
    debug = await say(client, container, "u", "s", "hello again", debug=True)
    assert plain["debug"] is None and "categories" in debug["debug"]


async def test_delete_memory_endpoint(client, container):
    await say(client, container, "u", "s", "I live in Pune.")
    assert (await client.delete("/users/u/memories/location")).json() == {"deleted": "location"}
    assert (await client.delete("/users/u/memories/location")).status_code == 404
    assert (await client.get("/users/u/memories")).json() == []


async def test_memory_endpoints_validate_the_user_id(client):
    assert (await client.get("/users/bad:id/memories")).status_code == 422


# ---- extraction rules ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "My friend speaks Hindi.",                      # mentions a language but states no preference
    "What is my sun sign?",
    "Saturn is in my seventh house, right?",
    "Thanks!",
])
def test_statements_that_look_similar_do_not_create_memories(text):
    assert rule_extract(candidate_statements(text)) == []


def test_a_language_word_in_another_context_is_not_a_language_preference():
    ops = {o.key: o.value for o in rule_extract(candidate_statements("I love English movies."))}
    assert ops == {"interests": "English movies"}


def test_hinglish_language_preference():
    ops = rule_extract(candidate_statements("Mujhe Hindi mein advice chahiye."))
    assert [(o.key, o.value) for o in ops] == [("language_preference", "Hindi")]


def test_business_and_promotion_goals_are_career_goals():
    for text, fragment in [
        ("I want to start my own business.", "start my own business"),
        ("I'm hoping to get a promotion this year.", "promotion"),
    ]:
        ops = {o.key: o.value for o in rule_extract(candidate_statements(text))}
        assert fragment in ops["career_goal"]


def test_financial_goal_needs_a_money_noun():
    assert {o.key for o in rule_extract(candidate_statements("I plan to buy a house in two years."))} == {"financial_goal"}
    assert rule_extract(candidate_statements("I want to buy a cake.")) == []


def test_many_facts_in_one_message():
    ops = {o.key: o.value for o in rule_extract(candidate_statements(
        "I'm a software engineer. I live in Pune and I love chess. I'm single."))}
    assert set(ops) == {"current_job", "location", "interests", "relationship_status"}
    assert ops["current_job"] == "software engineer" and ops["location"] == "Pune"


# ---- query keywords (English, Hinglish, Devanagari) -------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("मेरी नौकरी कैसी रहेगी", ["career"]),
    ("meri naukri ke baare mein batao", ["career"]),
    ("When will my shaadi happen?", ["relationships"]),
    ("Will my savings grow? Also my health.", ["finance", "health"]),
    ("Tell me about Saturn", []),
])
def test_category_keywords_across_languages(text, expected):
    assert detect_categories(text) == expected


def test_recall_and_followup_detection():
    assert has_recall_intent("yaad hai maine kya bataya tha?") and has_recall_intent("What did I tell you earlier?")
    assert not has_recall_intent("What should I focus on?")
    assert is_followup("and next year?") and is_followup("Why?") and not is_followup("Will I get a promotion this year?")


# ---- Ollama client (kept as the fallback provider) ---------------------------------------------------------------
def ollama(handler) -> OllamaClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama.test")
    return OllamaClient("http://ollama.test", "llama3.2:3b", client=http)


MESSAGES = [{"role": "user", "content": "hi"}]


async def test_ollama_chat_and_json_requests():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": '{"operations": []}'}})

    client = ollama(handler)
    assert await client.chat(MESSAGES) == '{"operations": []}'
    schema = {"type": "object"}
    assert await client.chat_json(MESSAGES, schema) == {"operations": []}
    assert seen[0]["stream"] is False and "format" not in seen[0]
    assert seen[1]["format"] == schema and seen[1]["options"]["temperature"] == 0.0


async def test_ollama_stream_parses_ndjson():
    lines = [{"message": {"content": "Hel"}, "done": False}, {"message": {"content": "lo"}, "done": False},
             {"message": {"content": ""}, "done": True}]
    body = "\n".join(json.dumps(x) for x in lines) + "\n"
    client = ollama(lambda r: httpx.Response(200, content=body.encode()))
    assert [t async for t in client.stream(MESSAGES)] == ["Hel", "lo"]


@pytest.mark.parametrize("response", [httpx.Response(500, text="boom"), httpx.Response(200, json={"message": {"content": ""}})])
async def test_ollama_errors_become_llm_errors(response):
    with pytest.raises(LLMError):
        await ollama(lambda r: response).chat(MESSAGES)
