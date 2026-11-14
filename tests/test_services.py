"""ChatService lifecycle, summariser fallbacks, prompt assembly and concurrency."""
from __future__ import annotations

import asyncio

from app.llm.base import LLMError
from app.llm.mock import MockLLM
from app.models import Memory, Message, RetrievalResult, RetrievedMemory
from app.services.prompt import build_messages
from app.services.summarizer import Summarizer
from tests.conftest import say


# ---- summariser fallbacks (LLM down) --------------------------------------------------------------
async def test_fold_falls_back_to_a_heuristic_when_the_llm_is_down():
    llm = MockLLM()
    llm.fail = True
    msgs = [Message(role="user", content="I want to switch jobs"), Message(role="assistant", content="Okay")]
    summary = await Summarizer(llm).fold("earlier notes", msgs)
    assert "earlier notes" in summary and "switch jobs" in summary          # the user's words survive
    assert "Okay" not in summary                                            # assistant filler does not


async def test_final_summary_falls_back_and_still_yields_topics():
    llm = MockLLM()
    llm.fail = True
    summary, topics = await Summarizer(llm).finalize("", [Message(role="user", content="my job and salary")])
    assert "job" in summary and topics == ["career"]


async def test_blank_llm_summary_triggers_the_fallback_too():
    llm = MockLLM(json_handler=lambda m, s: {"summary": "  ", "topics": []})
    summary, _ = await Summarizer(llm).finalize("", [Message(role="user", content="hello there")])
    assert "hello there" in summary


# ---- session lifecycle ---------------------------------------------------------------------------------
async def test_ending_an_empty_session_returns_nothing(client, container):
    assert await container.chat.end_session("u", "never-used") is None
    body = (await client.post("/users/u/sessions/never-used/end")).json()
    assert body == {"ended": False, "summary": None}


async def test_reusing_a_session_id_after_it_ended_keeps_the_earlier_summary(make_container):
    from httpx import ASGITransport, AsyncClient
    from app.main import create_app

    calls = {"n": 0}

    def handler(messages, schema):
        if "summary" in schema["properties"]:
            calls["n"] += 1
            return {"summary": f"summary number {calls['n']}", "topics": ["career"]}
        return {"operations": []}

    llm = MockLLM(json_handler=handler)
    container = make_container(llm=llm)
    async with AsyncClient(transport=ASGITransport(app=create_app(container)), base_url="http://t") as c:
        await say(c, container, "u", "s1", "hello one")
        await container.chat.end_session("u", "s1")
        await say(c, container, "u", "s1", "hello two")                       # same id comes back
        final = await container.chat.end_session("u", "s1")

    assert any("Earlier summary:\nsummary number 1" in call[1]["content"] for call in llm.json_calls)
    assert final.message_count == 4                                           # 2 turns across both runs
    assert (await container.memory_store.get_summary("u", "s1")).summary == "summary number 2"


async def test_sessions_of_the_same_user_do_not_share_history(client, container):
    await say(client, container, "u", "s1", "alpha secret")
    await say(client, container, "u", "s2", "beta question")
    prompt = container.llm.last_prompt
    assert [m["content"] for m in prompt[1:]] == ["beta question"]


# ---- concurrency + streaming ------------------------------------------------------------------------------
async def test_concurrent_messages_from_one_user_do_not_lose_memory_updates(client, container):
    await asyncio.gather(*[
        client.post("/chat", json={"user_id": "u", "session_id": "s", "message": f"I love {hobby}."})
        for hobby in ("cricket", "cooking", "chess")
    ])
    await container.chat.drain()
    interests = (await container.memory_store.get("u", "interests")).value
    assert {i.strip() for i in interests.split(",")} == {"cricket", "cooking", "chess"}


class DroppingStreamLLM(MockLLM):
    async def stream(self, messages, *, temperature=None):
        yield "Namaste "
        yield "friend"
        raise LLMError("connection dropped")


async def test_an_interrupted_stream_keeps_the_partial_answer(make_container):
    container = make_container(llm=DroppingStreamLLM())
    seen: list[str] = []

    async def on_token(token: str) -> None:
        seen.append(token)

    result = await container.chat.chat("u", "s", "hello", on_token=on_token)
    await container.chat.drain()
    assert seen == ["Namaste ", "friend"]
    assert result.response == "Namaste friend" and result.degraded is True


async def test_streaming_outage_before_any_token_sends_the_fallback_text(container):
    container.llm.fail = True
    seen: list[str] = []

    async def on_token(token: str) -> None:
        seen.append(token)

    result = await container.chat.chat("u", "s", "hello", on_token=on_token)
    await container.chat.drain()
    assert result.degraded and seen == [result.response]


# ---- prompt assembly --------------------------------------------------------------------------------------------
def _retrieved(key: str, value: str) -> RetrievedMemory:
    memory = Memory(user_id="u", key=key, value=value, category="career", kind="slot")
    return RetrievedMemory(memory=memory, score=1.0, similarity=0.5, reasons=["category"])


def test_prompt_contains_only_the_sections_that_have_content():
    bare = build_messages(RetrievalResult(), "", [], "hi")
    assert [m["role"] for m in bare] == ["system", "user"]
    for heading in ("## Known about the user", "## Notes from earlier sessions", "## Summary of this conversation"):
        assert heading not in bare[0]["content"]


def test_prompt_lists_memories_summary_and_history_in_order():
    retrieval = RetrievalResult(memories=[_retrieved("career_goal", "planning to switch jobs")])
    history = [Message(role="user", content="q1"), Message(role="assistant", content="a1")]
    messages = build_messages(retrieval, "talked about timing", history, "q2")

    system = messages[0]["content"]
    assert "## Known about the user\n- career goal: planning to switch jobs" in system
    assert "## Summary of this conversation so far\ntalked about timing" in system
    assert [(m["role"], m["content"]) for m in messages[1:]] == [("user", "q1"), ("assistant", "a1"), ("user", "q2")]
