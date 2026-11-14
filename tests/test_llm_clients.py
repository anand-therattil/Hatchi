"""The vLLM (OpenAI-compatible) client against a fake HTTP server - request shape, SSE streaming, errors."""
from __future__ import annotations

import json

import httpx
import pytest

from app.llm.base import LLMError
from app.llm.openai_compat import OpenAICompatClient

MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def make_client(handler, **kwargs) -> OpenAICompatClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://vllm.test")
    return OpenAICompatClient("http://vllm.test", "my-model", client=http, **kwargs)


def completion(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


async def test_chat_sends_an_openai_style_request_and_returns_the_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json=completion("  Namaste  "))

    reply = await make_client(handler, temperature=0.4, max_tokens=256).chat(MESSAGES)
    assert reply == "Namaste"
    assert seen["url"] == "http://vllm.test/v1/chat/completions"
    assert seen["body"]["model"] == "my-model" and seen["body"]["messages"] == MESSAGES
    assert seen["body"]["stream"] is False and seen["body"]["max_tokens"] == 256 and seen["body"]["temperature"] == 0.4


async def test_stream_parses_server_sent_events():
    events = [
        {"choices": [{"delta": {"role": "assistant"}}]},          # role-only first chunk: no content
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    tokens = [t async for t in make_client(handler).stream(MESSAGES)]
    assert tokens == ["Hel", "lo"]


async def test_chat_json_requests_schema_constrained_decoding():
    schema = {"type": "object", "properties": {"operations": {"type": "array"}}, "required": ["operations"]}
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion('{"operations": []}'))

    assert await make_client(handler).chat_json(MESSAGES, schema) == {"operations": []}
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["body"]["response_format"]["json_schema"]["schema"] == schema
    assert seen["body"]["temperature"] == 0.0                     # deterministic extraction


async def test_api_key_is_sent_as_a_bearer_token():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=completion("ok"))

    client = OpenAICompatClient("http://vllm.test", "m", api_key="secret")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vllm.test", headers=client._client.headers
    )
    await client.chat(MESSAGES)
    assert seen["auth"] == "Bearer secret"


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="CUDA out of memory"),
    httpx.Response(200, json={"choices": []}),                      # unexpected shape
    httpx.Response(200, json=completion("")),                       # empty reply
])
async def test_bad_server_responses_become_llm_errors(response):
    with pytest.raises(LLMError):
        await make_client(lambda request: response).chat(MESSAGES)


async def test_error_status_includes_the_server_message():
    with pytest.raises(LLMError, match="model not found"):
        await make_client(lambda r: httpx.Response(404, text="model not found")).chat(MESSAGES)


async def test_stream_error_status_becomes_llm_error():
    with pytest.raises(LLMError, match="503"):
        _ = [t async for t in make_client(lambda r: httpx.Response(503, text="overloaded")).stream(MESSAGES)]


async def test_invalid_json_from_chat_json_is_an_llm_error():
    with pytest.raises(LLMError, match="invalid JSON"):
        await make_client(lambda r: httpx.Response(200, json=completion("not json"))).chat_json(MESSAGES, {})


async def test_unreachable_server_is_an_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMError):
        await make_client(handler).chat(MESSAGES)
