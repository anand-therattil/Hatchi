"""Decides what from a user message is worth remembering long-term.

Pipeline:  gate (statements only)  ->  rules (high precision)  ->  LLM (fills gaps)
Rule results win on conflicts; LLM output is validated against the slot vocabulary and a
grounding check (its value must share a word with what the user actually said), which
guards against small models hallucinating "memories".
"""
from __future__ import annotations

import logging
from typing import Any

from app.llm.base import LLMClient, LLMError
from app.memory.categories import tokenize
from app.memory.rules import candidate_statements, rule_extract
from app.memory.schema import NEW_FACT_KEY, SLOT_SPECS, valid_key
from app.models import MemoryOp
from app.stores.base import MemoryStore

log = logging.getLogger("hatchi.extractor")

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["save", "delete"]},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "explicit": {"type": "boolean"},
                },
                "required": ["action", "key", "value", "explicit"],
            },
        }
    },
    "required": ["operations"],
}

_SYSTEM = """You extract long-term user memories for an astrology guidance app.
Read the user's statements and output JSON: {"operations": [...]}.

Each operation: {"action": "save" | "delete", "key": <one of the keys below>, "value": <short text>, "explicit": <bool>}.

Keys:
%(keys)s

Rules:
- Only save durable facts, goals, plans or preferences the user stated about THEMSELVES.
- Do NOT save questions, small talk, or things the assistant said.
- If the user corrects or withdraws something already known (see "Already known"), output a "save" with the
  same key and the new value, or a "delete" with that key if they withdraw it.
- "explicit" is true when the user said it outright, false when you only inferred it.
- value must be a short phrase copied or lightly condensed from the user's words.
- If nothing is worth remembering, output {"operations": []}.

Already known about this user:
%(known)s"""


def _grounded(value: str, text: str) -> bool:
    """Cheap hallucination guard: the saved value must share a real word with the user's text."""
    text_tokens = set(tokenize(text))
    return any(len(t) >= 3 and t in text_tokens for t in tokenize(value))


class MemoryExtractor:
    def __init__(self, llm: LLMClient, memory_store: MemoryStore):
        self._llm = llm
        self._store = memory_store

    async def extract(self, user_id: str, message: str) -> list[MemoryOp]:
        statements = candidate_statements(message)
        if not statements:
            return []
        text = " ".join(statements)

        rule_ops = rule_extract(statements)
        llm_ops = await self._llm_extract(user_id, text)

        merged: dict[tuple[str, str], MemoryOp] = {}
        for op in llm_ops:
            merged[(op.key, op.value if op.key == NEW_FACT_KEY else "")] = op
        for op in rule_ops:  # rules override the LLM for the same slot
            merged[(op.key, "")] = op
        return list(merged.values())

    async def _llm_extract(self, user_id: str, text: str) -> list[MemoryOp]:
        existing = await self._store.list(user_id)
        known = "\n".join(f"- {m.key}: {m.value}" for m in existing[:20]) or "(nothing yet)"
        keys = "\n".join(f"- {s.key}: {s.description}" for s in SLOT_SPECS.values())
        messages = [
            {"role": "system", "content": _SYSTEM % {"keys": keys, "known": known}},
            {"role": "user", "content": text},
        ]
        try:
            data = await self._llm.chat_json(messages, EXTRACTION_SCHEMA)
        except LLMError as exc:
            log.warning("LLM extraction unavailable, using rules only: %s", exc)
            return []
        return self._parse(data, text, {m.key for m in existing})

    @staticmethod
    def _parse(data: dict, text: str, existing_keys: set[str]) -> list[MemoryOp]:
        ops: list[MemoryOp] = []
        for raw in data.get("operations", []) if isinstance(data, dict) else []:
            try:
                action = raw["action"]
                key = str(raw["key"]).strip()
                value = str(raw.get("value", "")).strip()[:200]
                explicit = bool(raw.get("explicit", True))
            except (KeyError, TypeError):
                continue
            if action not in ("save", "delete") or not valid_key(key):
                continue
            if action == "delete":
                if key in existing_keys:
                    ops.append(MemoryOp(action="delete", key=key))
                continue
            if not value or not _grounded(value, text):
                continue
            ops.append(MemoryOp(action="save", key=key, value=value, explicit=explicit))
        return ops
