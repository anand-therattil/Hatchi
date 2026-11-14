"""Conversation summarisation: rolling (mid-session) and final (session end).

Both fall back to a deterministic heuristic when the LLM is unavailable, so a model outage
never loses conversation state.
"""
from __future__ import annotations

import logging
from typing import Any

from app.llm.base import LLMClient, LLMError
from app.memory.categories import detect_categories
from app.models import Message

log = logging.getLogger("hatchi.summarizer")

FINAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "topics"],
}

_FOLD_SYSTEM = (
    "You maintain a running summary of a chat between a user and an astrology guidance assistant. "
    "Merge the new messages into the current summary. Keep it under 120 words. Preserve the user's facts, "
    "goals, preferences, decisions and unanswered questions; drop pleasantries. Output only the summary."
)
_FINAL_SYSTEM = (
    "Summarise this finished chat between a user and an astrology guidance assistant for future recall. "
    'Output JSON {"summary": "<under 100 words, third person, focus on what the user shared and what was '
    'advised>", "topics": ["<1-4 short lowercase topics>"]}.'
)


def transcript(messages: list[Message]) -> str:
    return "\n".join(f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in messages)


def heuristic_summary(previous: str, messages: list[Message], limit: int = 700) -> str:
    """No-LLM fallback: keep what the user said, truncated."""
    said = "; ".join(m.content.strip().replace("\n", " ")[:160] for m in messages if m.role == "user")
    combined = " | ".join(p for p in (previous.strip(), f"User said: {said}" if said else "") if p)
    return combined[-limit:]


class Summarizer:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def fold(self, previous: str, messages: list[Message]) -> str:
        """Fold old messages into the rolling summary."""
        prompt = f"Current summary:\n{previous or '(none)'}\n\nNew messages:\n{transcript(messages)}\n\nUpdated summary:"
        try:
            text = await self._llm.chat(
                [{"role": "system", "content": _FOLD_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return text.strip() or heuristic_summary(previous, messages)
        except LLMError as exc:
            log.warning("fold fell back to heuristic: %s", exc)
            return heuristic_summary(previous, messages)

    async def finalize(self, previous: str, messages: list[Message]) -> tuple[str, list[str]]:
        """Produce the end-of-session summary and topic tags."""
        body = f"Earlier summary:\n{previous or '(none)'}\n\nRecent messages:\n{transcript(messages)}"
        try:
            data = await self._llm.chat_json(
                [{"role": "system", "content": _FINAL_SYSTEM}, {"role": "user", "content": body}],
                FINAL_SCHEMA,
            )
            summary = str(data.get("summary", "")).strip()
            topics = [str(t).strip().lower() for t in data.get("topics", []) if str(t).strip()][:4]
            if summary:
                return summary, topics
        except LLMError as exc:
            log.warning("final summary fell back to heuristic: %s", exc)
        summary = heuristic_summary(previous, messages)
        return summary, detect_categories(summary)
