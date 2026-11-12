"""EvalClient wiring for Anthropic prompt-cache markers."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.dialects.chat import ChatDialect

BASE_URL = "https://api.pinference.ai/api/v1"


def _body() -> dict:
    return {
        "model": "anthropic/claude-opus-4.8",
        "messages": [
            {"role": "system", "content": "stable instructions"},
            {"role": "user", "content": "dynamic rollout turn"},
            {"role": "assistant", "content": "working"},
            {"role": "user", "content": "continue"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {}},
            }
        ],
    }


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        content=json.dumps(
            {
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 0,
                "model": "anthropic/claude-opus-4.8",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
    )


class _StreamResponse:
    status_code = 200

    def __init__(self) -> None:
        self.headers = {"content-type": "text/event-stream"}

    async def aiter_bytes(self):
        yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_get_response_sends_configured_retention() -> None:
    client = EvalClient(EvalClientConfig(base_url=BASE_URL, cache_retention="long"))
    client._request = AsyncMock(return_value=_response())  # type: ignore[method-assign]
    body = _body()
    original = deepcopy(body)

    await client.get_response(ChatDialect(), body, SamplingConfig())

    sent = client._request.await_args.args[1]
    assert sent["tools"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    assert sent["messages"][-1]["content"][0]["cache_control"]["ttl"] == "1h"
    assert body == original


@pytest.mark.asyncio
async def test_relay_sends_configured_retention() -> None:
    client = EvalClient(EvalClientConfig(base_url=BASE_URL, cache_retention="long"))
    client._request = AsyncMock(return_value=_StreamResponse())  # type: ignore[method-assign]
    body = _body()
    original = deepcopy(body)

    await client.relay(ChatDialect(), body, session_id="trace-1")

    sent = client._request.await_args.args[1]
    assert sent["tools"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    assert body == original


def test_config_rejects_unknown_retention() -> None:
    with pytest.raises(ValidationError):
        EvalClientConfig(cache_retention="forever")  # type: ignore[arg-type]
