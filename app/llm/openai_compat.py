"""Client for any OpenAI-compatible chat server - in particular vLLM (`vllm serve ...`).

    POST {base_url}/v1/chat/completions
      - plain reply:   {"messages": [...]}
      - streaming:     "stream": true  -> Server-Sent Events ("data: {...}" ... "data: [DONE]")
      - structured:    "response_format": {"type": "json_schema", ...} -> decoding constrained to the schema
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import httpx

from app.llm.base import LLMError


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        timeout: float = 120.0,
        temperature: float = 0.4,
        max_tokens: int = 512,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=httpx.Timeout(timeout, connect=5.0)
        )

    def _payload(
        self, messages: list[dict], *, stream: bool, temperature: Optional[float], max_tokens: Optional[int] = None
    ) -> dict:
        return {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }

    @staticmethod
    def _describe(exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            return f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        return str(exc) or exc.__class__.__name__

    async def _post(self, payload: dict) -> dict:
        try:
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"LLM request failed: {self._describe(exc)}") from exc

    @staticmethod
    def _content(data: dict) -> str:
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response shape: {str(data)[:200]}") from exc

    async def chat(self, messages: list[dict], *, temperature: Optional[float] = None) -> str:
        content = self._content(await self._post(self._payload(messages, stream=False, temperature=temperature)))
        if not content:
            raise LLMError("LLM returned an empty reply")
        return content

    async def stream(self, messages: list[dict], *, temperature: Optional[float] = None) -> AsyncIterator[str]:
        payload = self._payload(messages, stream=True, temperature=temperature)
        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    choices = json.loads(data).get("choices") or []
                    token = (choices[0].get("delta") or {}).get("content") if choices else None
                    if token:
                        yield token
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"LLM stream failed: {self._describe(exc)}") from exc

    async def chat_json(self, messages: list[dict], schema: dict) -> dict:
        payload = self._payload(messages, stream=False, temperature=0.0, max_tokens=400)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": schema, "strict": True},
        }
        content = self._content(await self._post(payload))
        try:
            parsed = json.loads(content)
        except ValueError as exc:
            raise LLMError(f"LLM returned invalid JSON: {content[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("LLM JSON reply was not an object")
        return parsed

    async def aclose(self) -> None:
        await self._client.aclose()
