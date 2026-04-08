import json
from typing import Any

import httpx
import pytest
from renderers.base import ParsedResponse
from renderers.proxy import RenderingProxy, UPSTREAM_BASE_URL_HEADER


class _FakeToolCallRenderer:
    def parse_response(self, completion_ids: list[int]) -> ParsedResponse:
        assert completion_ids == [101, 102]
        return ParsedResponse(
            content="",
            reasoning_content="I should use a tool.",
            tool_calls=[
                {
                    "function": {
                        "name": "echo",
                        "arguments": {"text": "hello"},
                    }
                }
            ],
        )


def test_proxy_converts_parsed_tool_call_to_openai_chat_response():
    proxy = object.__new__(RenderingProxy)
    proxy._renderer = _FakeToolCallRenderer()

    response = RenderingProxy._to_chat_response(
        proxy,
        {
            "id": "gen_1",
            "model": "test-model",
            "choices": [
                {
                    "token_ids": [101, 102],
                    "logprobs": [-0.1, -0.2],
                    "finish_reason": "tool_calls",
                    "routed_experts": [[[1]], [[2]]],
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
        [1, 2, 3],
    )

    choice: dict[str, Any] = response["choices"][0]
    assert response["prompt_token_ids"] == [1, 2, 3]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["token_ids"] == [101, 102]
    assert choice["routed_experts"] == [[[1]], [[2]]]
    assert choice["message"]["reasoning_content"] == "I should use a tool."
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"text": "hello"}'},
        }
    ]
    assert choice["logprobs"] == {
        "content": [
            {"token": "", "logprob": -0.1},
            {"token": "", "logprob": -0.2},
        ]
    }


class _FakeRenderer:
    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        assert add_generation_prompt is True
        return [1, 2, 3]

    def get_stop_token_ids(self):
        return [99]

    def parse_response(self, completion_ids: list[int]) -> ParsedResponse:
        return ParsedResponse(content="done")


class _ToollessRenderer:
    supports_tools = False

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        raise AssertionError("render_ids should not be called")

    def get_stop_token_ids(self):
        raise AssertionError("get_stop_token_ids should not be called")

    def parse_response(self, completion_ids: list[int]) -> ParsedResponse:
        raise AssertionError("parse_response should not be called")


@pytest.mark.asyncio
async def test_proxy_routes_to_selected_upstream_and_forwards_headers():
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "gen_1",
                "model": "test-model",
                "choices": [{"token_ids": [4, 5], "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    proxy = RenderingProxy(_FakeRenderer(), vllm_base_url="http://default:8000")
    await proxy._client.aclose()
    proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=600.0)

    transport = httpx.ASGITransport(app=proxy.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 7},
            headers={
                UPSTREAM_BASE_URL_HEADER: "http://backend.example:8000/v1",
                "Authorization": "Bearer secret",
                "X-Session-ID": "session-123",
            },
        )

    await proxy.close()

    assert response.status_code == 200
    assert captured["url"] == "http://backend.example:8000/v1/generate"
    assert captured["headers"]["authorization"] == "Bearer secret"
    assert captured["headers"]["x-session-id"] == "session-123"
    assert UPSTREAM_BASE_URL_HEADER.lower() not in captured["headers"]
    assert captured["body"]["prompt_token_ids"] == [1, 2, 3]
    assert captured["body"]["max_tokens"] == 7
    assert captured["body"]["stop_token_ids"] == [99]


@pytest.mark.asyncio
async def test_proxy_rejects_tools_when_renderer_does_not_support_them():
    proxy = RenderingProxy(_ToollessRenderer(), vllm_base_url="http://default:8000")
    transport = httpx.ASGITransport(app=proxy.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}],
            },
        )

    await proxy.close()

    assert response.status_code == 400
    assert "does not support tools" in response.text
