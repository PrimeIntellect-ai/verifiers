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
async def test_completions_request_with_prebuilt_prompt_ids_still_extracts_images():
    client = _FakeClient()
    image_url = "data:image/png;base64,aGVsbG8="

    await completions_request(
        client=client,
        renderer=_NoRenderRenderer(),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        model="test-model",
        prompt_ids=[11, 12, 13],
    )

    assert client.calls[0]["body"]["images"] == [
        {"data": "aGVsbG8=", "media_type": "image/png"}
    ]


class _ExpandingClient(_FakeClient):
    """Fake server that echoes back a server-expanded prompt (e.g. vLLM
    replacing a single <|image_pad|> with the actual patch-grid run)."""

    async def post(self, path, *, cast_to=dict, body=None):
        base = await super().post(path, cast_to=cast_to, body=body)
        sent = body.get("prompt_token_ids") if body else []
        # Simulate vLLM expanding one placeholder token (id=2) into 4 copies
        expanded = []
        for tok in sent:
            if tok == 2:
                expanded.extend([2, 2, 2, 2])
            else:
                expanded.append(tok)
        base["prompt_token_ids"] = expanded
        return base


@pytest.mark.asyncio
async def test_completions_request_returns_server_expanded_prompt_ids():
    client = _ExpandingClient()

    result = await completions_request(
        client=client,
        renderer=_FakeRenderer(),
        messages=[{"role": "user", "content": "hi"}],
        model="test-model",
        tools=[{"type": "function", "function": {"name": "echo"}}],
    )

    # Renderer sends [1, 2, 3]; fake server expanded the image-pad-like
    # token 2 into four copies. The client must surface the server's
    # expanded form so the trainer replays the exact tokens vLLM ran on.
    assert client.calls[0]["body"]["prompt_token_ids"] == [1, 2, 3]
    assert result["prompt_ids"] == [1, 2, 2, 2, 2, 3]
