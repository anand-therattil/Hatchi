"""A scriptable fake LLM for tests and infrastructure-free demos. Records every call it receives."""
from __future__ import annotations

from typing import AsyncIterator, Callable, Optional

from app.llm.base import LLMError


def _default_reply(messages: list[dict]) -> str:
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return f"[mock] You said: {last_user}"


def _default_json(messages: list[dict], schema: dict) -> dict:
    props = schema.get("properties", {})
    if "operations" in props:
        return {"operations": []}
    if "summary" in props:
        return {"summary": "[mock] session summary", "topics": []}
    return {}


class MockLLM:
    def __init__(
        self,
        reply: Optional[Callable[[list[dict]], str]] = None,
        json_handler: Optional[Callable[[list[dict], dict], dict]] = None,
    ):
        self.reply = reply or _default_reply
        self.json_handler = json_handler or _default_json
        self.chat_calls: list[list[dict]] = []   # every prompt sent for a user-facing reply
        self.json_calls: list[list[dict]] = []   # every structured-output prompt
        self.fail = False                        # flip to simulate an outage

    async def chat(self, messages: list[dict], *, temperature: Optional[float] = None) -> str:
        if self.fail:
            raise LLMError("mock LLM unavailable")
        self.chat_calls.append(messages)
        return self.reply(messages)

    async def stream(self, messages: list[dict], *, temperature: Optional[float] = None) -> AsyncIterator[str]:
        if self.fail:
            raise LLMError("mock LLM unavailable")
        self.chat_calls.append(messages)
        text = self.reply(messages)
        for word in text.split(" "):
            yield word + " "

    async def chat_json(self, messages: list[dict], schema: dict) -> dict:
        if self.fail:
            raise LLMError("mock LLM unavailable")
        self.json_calls.append(messages)
        return self.json_handler(messages, schema)

    @property
    def last_prompt(self) -> list[dict]:
        return self.chat_calls[-1]

    @property
    def last_system_prompt(self) -> str:
        return self.chat_calls[-1][0]["content"]
