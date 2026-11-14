"""LLM interface. Swap providers (or mock) by implementing these three methods."""
from __future__ import annotations

from typing import AsyncIterator, Optional, Protocol


class LLMError(Exception):
    """Raised when the model is unreachable, times out or returns unusable output."""


class LLMClient(Protocol):
    async def chat(self, messages: list[dict], *, temperature: Optional[float] = None) -> str:
        """Return the full assistant reply."""

    def stream(self, messages: list[dict], *, temperature: Optional[float] = None) -> AsyncIterator[str]:
        """Yield the reply token-by-token (an async generator)."""

    async def chat_json(self, messages: list[dict], schema: dict) -> dict:
        """Return a JSON object constrained to `schema`."""
