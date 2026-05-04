import asyncio
import base64

import numpy as np

from renderers.base import ParsedResponse
from renderers.client import generate


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
    """Mocks AsyncOpenAI's `.post()`. The renderer client builds an absolute
    URL off ``client.base_url``, so we expose one that includes the /v1 suffix
    the OpenAI SDK normally appends."""

    def __init__(self):
        self.calls = []
        self.base_url = "http://fake-host:8000/v1"

    async def post(self, path, *, cast_to=dict, body=None, options=None):
        self.calls.append(
            {"path": path, "cast_to": cast_to, "body": body, "options": options}
        )
        routed_experts = np.array([[[1]], [[2]]], dtype=np.int32)
        return {
            "request_id": "gen-test",
            "choices": [
                {
                    "index": 0,
                    "token_ids": [7, 8],
                    "logprobs": {
                        "content": [
                            {"token": "token_id:7", "logprob": -0.1},
                            {"token": "token_id:8", "logprob": -0.2},
                        ]
                    },
                    "finish_reason": "stop",
                    "routed_experts": {
                        "data": base64.b85encode(routed_experts.tobytes()).decode(
                            "ascii"
                        ),
                        "shape": list(routed_experts.shape),
                    },
                }
            ],
        }


def test_generate_builds_request_body_and_parses_response():
    client = _FakeClient()

    result = asyncio.run(
        generate(
            client=client,
            renderer=_FakeRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "echo"}}],
            sampling_params={"temperature": 0.3, "max_tokens": 7, "min_tokens": 2},
            cache_salt="ckpt-42",
        )
    )

    assert len(client.calls) == 1
    # /inference/v1/generate is mounted at the server root, so we post to
    # an absolute URL stripped of the OpenAI SDK's automatic /v1 prefix.
    assert client.calls[0]["path"] == "http://fake-host:8000/inference/v1/generate"
    assert client.calls[0]["body"] == {
        "model": "test-model",
        "token_ids": [1, 2, 3],
        "cache_salt": "ckpt-42",
        "sampling_params": {
            "temperature": 0.3,
            "max_tokens": 7,
            "min_tokens": 2,
            "stop_token_ids": [99],
            "logprobs": 1,
            "skip_special_tokens": False,
        },
    }
    # finish_reason promoted from "stop" → "tool_calls" because the renderer
    # extracted tool calls client-side.
    assert result == {
        "request_id": "gen-test",
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
        "routed_experts": [[[1]], [[2]]],
    }


class _NoRenderRenderer(_FakeRenderer):
    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        raise AssertionError("prebuilt prompt ids should skip render_ids")


def test_generate_uses_prebuilt_prompt_ids_without_rendering():
    client = _FakeClient()

    result = asyncio.run(
        generate(
            client=client,
            renderer=_NoRenderRenderer(),
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            prompt_ids=[11, 12, 13],
        )
    )

    assert client.calls[0]["body"]["token_ids"] == [11, 12, 13]
    assert result["prompt_ids"] == [11, 12, 13]
