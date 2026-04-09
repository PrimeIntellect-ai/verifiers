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
    def __init__(self):
        self.calls = []

    async def post(self, path, *, cast_to=dict, body=None):
        self.calls.append({"path": path, "cast_to": cast_to, "body": body})
        routed_experts = np.array([[[1]], [[2]]], dtype=np.int32)
        return {
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "choices": [
                {
                    "token_ids": [7, 8],
                    "logprobs": [-0.1, -0.2],
                    "finish_reason": "tool_calls",
                    "routed_experts": {
                        "data": base64.b85encode(routed_experts.tobytes()).decode("ascii"),
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
