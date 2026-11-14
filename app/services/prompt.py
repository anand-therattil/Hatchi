"""Builds the message list sent to the LLM from the selected context."""
from __future__ import annotations

from app.models import Message, RetrievalResult

SYSTEM_PROMPT = """You are Naksh, a warm and practical astrology guidance assistant for MyNaksh.
Give thoughtful, grounded guidance. This is a demo without a real chart engine, so speak in general terms
(themes, timing, mindset) and never invent specific planetary positions or dates. Keep replies concise
(about 120 words) unless the user asks for more.

Personalisation rules:
- Use the facts under "Known about the user" only when they are relevant to the question. Never mention
  facts that are not listed there.
- If the user asks what you remember, answer strictly from the listed facts and earlier-session notes.
  If nothing relevant is listed, say you have nothing saved about that yet.
- If a preferred language is listed, reply in that language (Hindi in Devanagari script unless the user
  writes in Hinglish). Otherwise reply in the language the user writes in."""


def build_messages(
    retrieval: RetrievalResult,
    rolling_summary: str,
    recent: list[Message],
    user_message: str,
) -> list[dict]:
    system = SYSTEM_PROMPT

    if retrieval.memories:
        lines = "\n".join(f"- {r.memory.text}" for r in retrieval.memories)
        system += f"\n\n## Known about the user\n{lines}"
    if retrieval.past_sessions:
        lines = "\n".join(f"- {p.summary}" for p in retrieval.past_sessions)
        system += f"\n\n## Notes from earlier sessions\n{lines}"
    if rolling_summary:
        system += f"\n\n## Summary of this conversation so far\n{rolling_summary}"

    messages: list[dict] = [{"role": "system", "content": system}]
    messages += [{"role": m.role, "content": m.content} for m in recent]
    messages.append({"role": "user", "content": user_message})
    return messages
