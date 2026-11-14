"""End-to-end behaviour through the HTTP API (mock LLM, in-memory stores, real Chroma index)."""
from __future__ import annotations

import pytest

from app.llm.mock import MockLLM
from tests.conftest import say

JOB_MSG = "I'm planning to switch jobs next year, and I prefer advice in Hindi."


async def test_send_message_and_receive_response(client, container):
    data = await say(client, container, "user-1", "s1", "Hello there")
    assert data["user_id"] == "user-1" and data["session_id"] == "s1"
    assert data["response"].startswith("[mock]")
    assert data["context_used"] == []          # nothing known yet, nothing injected
    assert data["degraded"] is False


async def test_memory_saved_then_retrieved_in_a_new_session(client, container):
    await say(client, container, "user-1", "session-A", JOB_MSG)

    saved = {m["key"]: m["value"] for m in (await client.get("/users/user-1/memories")).json()}
    assert saved["language_preference"] == "Hindi"
    assert "switch jobs" in saved["career_goal"]
    assert len(saved) == 2                      # the message itself is NOT stored as a memory

    # brand-new session, same user
    data = await say(client, container, "user-1", "session-B", "What do you remember about my career plans?")
    assert "career_goal" in data["context_used"]
    assert "language_preference" in data["context_used"]
    assert "switch jobs next year" in container.llm.last_system_prompt   # it reached the LLM


async def test_recent_history_supports_followup_question(client, container):
    await say(client, container, "user-1", "s1", "What should I focus on for my career?")
    await say(client, container, "user-1", "s1", "Why do you say that?")

    prompt = container.llm.last_prompt
    assert [m["role"] for m in prompt] == ["system", "user", "assistant", "user"]
    assert prompt[1]["content"] == "What should I focus on for my career?"
    assert prompt[2]["content"].startswith("[mock]")
    assert prompt[3]["content"] == "Why do you say that?"


async def test_irrelevant_memories_are_excluded_from_llm_context(client, container):
    await say(client, container, "user-1", "s1", JOB_MSG)
    await say(client, container, "user-1", "s1", "I love cricket and cooking.")
    memories = {m["key"] for m in (await client.get("/users/user-1/memories")).json()}
    assert {"career_goal", "language_preference", "interests"} <= memories

    data = await say(client, container, "user-1", "s2", "What should I focus on for my career?", debug=True)

    assert "career_goal" in data["context_used"]
    assert "language_preference" in data["context_used"]      # standing preference is always on
    assert "interests" not in data["context_used"]
    assert "cricket" not in container.llm.last_system_prompt
    assert "interests" in data["debug"]["excluded"]


async def test_user_correction_updates_the_memory_in_place(client, container):
    await say(client, container, "user-1", "s1", "I prefer advice in Hindi.")
    await say(client, container, "user-1", "s1", "Actually, I prefer advice in English.")

    memories = (await client.get("/users/user-1/memories")).json()
    assert len(memories) == 1                                  # updated, not duplicated
    lang = memories[0]
    assert lang["value"] == "English" and lang["version"] == 2
    assert lang["history"][0]["value"] == "Hindi"              # old value archived

    await say(client, container, "user-1", "s2", "What should I focus on for my career?")
    prompt = container.llm.last_system_prompt                  # (the static rules text itself mentions Hindi)
    assert "- preferred language: English" in prompt
    assert "preferred language: Hindi" not in prompt
    hits = await container.vectors.search_memories("user-1", "preferred language", k=5)
    assert [h.document for h in hits] == ["preferred language: English"]   # index holds only the new version


async def test_withdrawing_a_plan_deletes_the_memory(client, container):
    await say(client, container, "user-1", "s1", "I'm planning to switch jobs next year.")
    assert any(m["key"] == "career_goal" for m in (await client.get("/users/user-1/memories")).json())

    await say(client, container, "user-1", "s1", "I'm not planning to switch jobs anymore.")
    assert (await client.get("/users/user-1/memories")).json() == []


async def test_memories_are_isolated_per_user(client, container):
    await say(client, container, "alice", "s1", JOB_MSG)
    data = await say(client, container, "bob", "s1", "What do you remember about my career plans?")
    assert data["context_used"] == []
    assert (await client.get("/users/bob/memories")).json() == []


async def test_finished_session_is_summarised_and_recalled_in_a_later_session(make_container):
    llm = MockLLM(json_handler=lambda m, schema: (
        {"summary": "User is planning to switch jobs next year and wants Hindi advice.", "topics": ["career"]}
        if "summary" in schema["properties"] else {"operations": []}
    ))
    container = make_container(llm=llm)
    from httpx import ASGITransport, AsyncClient
    from app.main import create_app

    async with AsyncClient(transport=ASGITransport(app=create_app(container)), base_url="http://t") as c:
        await say(c, container, "user-1", "session-A", JOB_MSG)
        ended = (await c.post("/users/user-1/sessions/session-A/end")).json()
        assert ended["ended"] is True and "switch jobs" in ended["summary"]["summary"]
        assert await container.sessions.recent("user-1", "session-A") == []   # Redis state cleared

        data = await say(c, container, "user-1", "session-B", "What did we discuss earlier about my job plans?")
        assert "session_summary:session-A" in data["context_used"]
        assert "Notes from earlier sessions" in llm.last_system_prompt


async def test_rolling_summary_keeps_the_history_window_bounded(make_container):
    from httpx import ASGITransport, AsyncClient
    from app.main import create_app

    container = make_container(history_max_messages=4, history_fold_count=2)
    async with AsyncClient(transport=ASGITransport(app=create_app(container)), base_url="http://t") as c:
        for i in range(5):
            await say(c, container, "user-1", "s1", f"message number {i}")

    recent = await container.sessions.recent("user-1", "s1")
    assert len(recent) <= 4
    assert await container.sessions.get_summary("user-1", "s1")         # old turns live on as a summary
    # (fold prompts also go through llm.chat, so look at every system prompt that was sent)
    assert any("Summary of this conversation so far" in call[0]["content"] for call in container.llm.chat_calls)


async def test_llm_outage_degrades_gracefully_but_still_remembers(client, container):
    container.llm.fail = True
    data = await say(client, container, "user-1", "s1", "I prefer advice in Hindi, please.")
    assert data["degraded"] is True and "trouble" in data["response"]

    memories = {m["key"]: m["value"] for m in (await client.get("/users/user-1/memories")).json()}
    assert memories == {"language_preference": "Hindi"}                 # rules still worked without the LLM
    history = await container.sessions.recent("user-1", "s1")
    assert [m.role for m in history] == ["user"]                         # canned reply not stored as a turn


async def test_idle_sessions_are_finalised_by_the_sweeper(client, container):
    await say(client, container, "user-1", "s1", "I prefer advice in Hindi.")
    container.sessions.force_last_active("user-1", "s1", 0.0)

    assert await container.chat.sweep_idle() == 1
    assert await container.memory_store.get_summary("user-1", "s1") is not None


@pytest.mark.parametrize("payload", [
    {"user_id": "u", "session_id": "s", "message": "   "},
    {"user_id": "u", "session_id": "s", "message": "x" * 4001},
    {"user_id": "bad:id", "session_id": "s", "message": "hi"},
    {"user_id": "u", "message": "hi"},
])
async def test_invalid_requests_are_rejected(client, payload):
    assert (await client.post("/chat", json=payload)).status_code == 422
