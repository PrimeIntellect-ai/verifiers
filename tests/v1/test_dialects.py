import json

import httpx
import pytest

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.types import SamplingConfig


def test_responses_reasoning_effort_overrides_codex_request() -> None:
    body = {
        "input": "Solve this.",
        "reasoning": {"effort": "medium", "summary": "auto"},
    }

    request = ResponsesDialect().apply_overrides(
        body,
        "openai/gpt-5.4",
        SamplingConfig(reasoning_effort="low"),
    )

    assert request["reasoning"] == {"effort": "low", "summary": "auto"}


def test_responses_reasoning_effort_is_omitted_when_unset() -> None:
    body = {
        "input": "Solve this.",
        "reasoning": {"effort": "medium", "summary": "auto"},
    }

    request = ResponsesDialect().apply_overrides(
        body,
        "openai/gpt-5.4",
        SamplingConfig(),
    )

    assert request["reasoning"] == {"effort": "medium", "summary": "auto"}


@pytest.mark.asyncio
async def test_codex_streaming_request_uses_eval_reasoning_effort() -> None:
    captured: dict = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
        )

    client = EvalClient("https://example.test/v1", "secret")
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        reply = await client.relay(
            ResponsesDialect(),
            {
                "input": "Solve this.",
                "stream": True,
                "reasoning": {"effort": "medium", "summary": "auto"},
            },
            "openai/gpt-5.4",
            SamplingConfig(reasoning_effort="low"),
        )
        assert b"".join([chunk async for chunk in reply.chunks]) == b"data: [DONE]\n\n"
    finally:
        await client.close()

    assert captured["reasoning"] == {"effort": "low", "summary": "auto"}


def test_anthropic_reasoning_effort_overrides_output_config() -> None:
    body = {
        "messages": [{"role": "user", "content": "Solve this."}],
        "max_tokens": 1024,
        "output_config": {"format": "compact"},
    }

    request = AnthropicDialect().apply_overrides(
        body,
        "claude-sonnet-4-6",
        SamplingConfig(reasoning_effort="low"),
    )

    assert request["output_config"] == {"format": "compact", "effort": "low"}
    assert "thinking" not in request


def test_anthropic_reasoning_effort_preserves_explicit_thinking() -> None:
    body = {
        "messages": [{"role": "user", "content": "Solve this."}],
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 512},
    }

    request = AnthropicDialect().apply_overrides(
        body,
        "claude-sonnet-4-6",
        SamplingConfig(reasoning_effort="high"),
    )

    assert request["output_config"] == {"effort": "high"}
    assert request["thinking"] == {"type": "enabled", "budget_tokens": 512}


@pytest.mark.asyncio
async def test_anthropic_streaming_request_uses_eval_reasoning_effort() -> None:
    captured: dict = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
        )

    client = EvalClient("https://api.anthropic.test", "secret")
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        reply = await client.relay(
            AnthropicDialect(),
            {
                "messages": [{"role": "user", "content": "Solve this."}],
                "max_tokens": 1024,
                "stream": True,
            },
            "claude-opus-4-6",
            SamplingConfig(reasoning_effort="low"),
        )
        assert b"".join([chunk async for chunk in reply.chunks]) == b"data: [DONE]\n\n"
    finally:
        await client.close()

    assert captured["output_config"] == {"effort": "low"}
    assert "thinking" not in captured
