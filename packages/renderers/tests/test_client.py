import base64

import numpy as np
import pytest

from renderers.base import ParsedResponse
from renderers.client import completions_request


class _FakeRenderer:
    supports_tools = True

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        assert messages == [{"role": "user", "content": "hi"}]
        assert tools == [{"type": "function", "function": {"name": "echo"}}]
        assert add_generation_prompt is True
        return [1, 2, 3]

    def get_stop_token_ids(self):
        return [99]

    def parse_response(self, completion_ids: list[int]) -> ParsedResponse:
        assert completion_ids == [7, 8]
        return ParsedResponse(
            content="done",
            reasoning_content="think",
            tool_calls=[
                {
                    "function": {
                        "name": "echo",
                        "arguments": {"text": "hello"},
                    }
                }
            ],
        )


class _FakeClient:
    def __init__(self, response=None):
        self.calls = []
        self.response = response

    async def post(self, path, *, cast_to=dict, body=None):
        self.calls.append({"path": path, "cast_to": cast_to, "body": body})
        if self.response is not None:
            return self.response
        routed_experts = np.array([[[1]], [[2]]], dtype=np.int32)
        return {
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "choices": [
                {
                    "token_ids": [7, 8],
                    "logprobs": [-0.1, -0.2],
                    "finish_reason": "tool_calls",
                    "routed_experts": {
                        "data": base64.b85encode(routed_experts.tobytes()).decode(
                            "ascii"
                        ),
                        "shape": list(routed_experts.shape),
                    },
                }
            ],
        }


@pytest.mark.asyncio
async def test_completions_request_builds_generate_body_and_parses_response():
    client = _FakeClient()

    result = await completions_request(
        client=client,
        renderer=_FakeRenderer(),
        messages=[{"role": "user", "content": "hi"}],
        model="test-model",
        tools=[{"type": "function", "function": {"name": "echo"}}],
        temperature=0.3,
        max_completion_tokens=7,
        extra_body={"min_tokens": 2},
    )

    assert len(client.calls) == 1
    assert client.calls[0]["path"] == "/generate"
    assert client.calls[0]["body"] == {
        "model": "test-model",
        "prompt_token_ids": [1, 2, 3],
        "stop_token_ids": [99],
        "temperature": 0.3,
        "max_tokens": 7,
        "min_tokens": 2,
    }
    assert result == {
        "id": "",
        "created": 0,
        "model": "",
        "prompt_ids": [1, 2, 3],
        "completion_ids": [7, 8],
        "completion_logprobs": [-0.1, -0.2],
        "content": "done",
        "reasoning_content": "think",
        "tool_calls": [
            {
                "function": {
                    "name": "echo",
                    "arguments": {"text": "hello"},
                }
            }
        ],
        "finish_reason": "tool_calls",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "routed_experts": [[[1]], [[2]]],
    }


class _NoRenderRenderer(_FakeRenderer):
    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        raise AssertionError("prebuilt prompt ids should skip render_ids")


@pytest.mark.asyncio
async def test_completions_request_uses_prebuilt_prompt_ids_without_rendering():
    client = _FakeClient()

    result = await completions_request(
        client=client,
        renderer=_NoRenderRenderer(),
        messages=[{"role": "user", "content": "hi"}],
        model="test-model",
        prompt_ids=[11, 12, 13],
    )

    assert client.calls[0]["body"]["prompt_token_ids"] == [11, 12, 13]
    assert result["prompt_ids"] == [11, 12, 13]


@pytest.mark.asyncio
async def test_completions_request_can_use_dynamo_chat_nvext_transport():
    client = _FakeClient(
        {
            "id": "chatcmpl-test",
            "model": "test-model",
            "nvext": {"completion_token_ids": [7, 8]},
            "choices": [
                {
                    "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                    "finish_reason": "stop",
                }
            ],
        }
    )

    result = await completions_request(
        client=client,
        renderer=_FakeRenderer(),
        messages=[{"role": "user", "content": "hi"}],
        model="test-model",
        tools=[{"type": "function", "function": {"name": "echo"}}],
        transport="dynamo_chat_nvext",
        temperature=0.3,
        max_completion_tokens=7,
        priority=4,
        extra_body={"min_tokens": 2},
    )

    assert client.calls[0]["path"] == "/chat/completions"
    body = client.calls[0]["body"]
    assert body["nvext"] == {
        "token_data": [1, 2, 3],
        "extra_fields": ["completion_token_ids"],
        "agent_hints": {"priority": 4},
    }
    assert body["messages"] == [{"role": "user", "content": "(token-in mode)"}]
    assert body["logprobs"] is True
    assert body["stop_token_ids"] == [99]
    assert body["max_completion_tokens"] == 7
    assert body["temperature"] == 0.3
    assert body["min_tokens"] == 2
    assert result["id"] == "chatcmpl-test"
    assert result["model"] == "test-model"
    assert result["completion_ids"] == [7, 8]
    assert result["completion_logprobs"] == [-0.1, -0.2]
    assert result["finish_reason"] == "tool_calls"
