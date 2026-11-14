"""Ollama provider (Llama 3.2 3B, Qwen2.5 1.5B, ... any model Ollama can serve)."""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import httpx

from app.llm.base import LLMError


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        temperature: float = 0.4,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self._model = model
        self._temperature = temperature
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(timeout, connect=5.0)
        )

    def _payload(self, messages: list[dict], *, stream: bool, temperature: Optional[float]) -> dict:
        return {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "keep_alive": "30m",
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_ctx": 4096,
            },
        }

    async def _post(self, payload: dict) -> dict:
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

    async def chat(self, messages: list[dict], *, temperature: Optional[float] = None) -> str:
        data = await self._post(self._payload(messages, stream=False, temperature=temperature))
        content = (data.get("message") or {}).get("content", "").strip()
        if not content:
            raise LLMError("Ollama returned an empty reply")
        return content

    async def stream(self, messages: list[dict], *, temperature: Optional[float] = None) -> AsyncIterator[str]:
        payload = self._payload(messages, stream=True, temperature=temperature)
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = (chunk.get("message") or {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"Ollama stream failed: {exc}") from exc

    async def chat_json(self, messages: list[dict], schema: dict) -> dict:
        payload = self._payload(messages, stream=False, temperature=0.0)
        payload["format"] = schema  # structured outputs: constrains decoding to the schema
        data = await self._post(payload)
        content = (data.get("message") or {}).get("content", "")
        try:
            parsed = json.loads(content)
        except ValueError as exc:
            raise LLMError(f"Ollama returned invalid JSON: {content[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("Ollama JSON reply was not an object")
        return parsed

    async def aclose(self) -> None:
        await self._client.aclose()
