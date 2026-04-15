from unittest.mock import patch

import pytest

import verifiers as vf
from verifiers.errors import EmptyModelResponseError
from renderers import RendererPool
from renderers.base import ParsedResponse
from verifiers.clients.renderer_client import (
    RendererClient,
    _get_incremental_prompt_ids,
    _is_valid_incremental_tail,
    _to_renderer_message,
)
from verifiers.types import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def test_renderer_client_honors_configured_renderer_name():
    RendererClient._shared_pools.clear()

    client = object.__new__(RendererClient)
    client._renderer = None
    client._pool_size = 1
    client._config = vf.ClientConfig(client_type="renderer", renderer="qwen3_vl")

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=object()
        ) as tokenizer_mock,
        patch(
            "verifiers.clients.renderer_client.create_renderer", return_value="renderer"
        ) as create_renderer_mock,
    ):
        pool = client._get_renderer_or_pool("Qwen/Qwen3-VL-4B-Instruct")

    assert isinstance(pool, RendererPool)
    tokenizer_mock.assert_called_once_with(
        "Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True
    )
    create_renderer_mock.assert_called_once_with(
        tokenizer_mock.return_value, renderer="qwen3_vl"
    )


def test_renderer_client_uses_renderer_model_name_override():
    RendererClient._shared_pools.clear()

    client = object.__new__(RendererClient)
    client._renderer = None
    client._pool_size = 1
    client._config = vf.ClientConfig(
        client_type="renderer",
        renderer="qwen3_vl",
        renderer_model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=object()
        ) as tokenizer_mock,
        patch(
            "verifiers.clients.renderer_client.create_renderer", return_value="renderer"
        ) as create_renderer_mock,
    ):
        pool = client._get_renderer_or_pool("r8-smoke")

    assert isinstance(pool, RendererPool)
    tokenizer_mock.assert_called_once_with(
        "Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True
    )
    create_renderer_mock.assert_called_once_with(
        tokenizer_mock.return_value, renderer="qwen3_vl"
    )


@pytest.mark.asyncio
async def test_renderer_client_accepts_dict_native_response_with_content():
    client = object.__new__(RendererClient)

    await client.raise_from_native_response({"content": "done"})


@pytest.mark.asyncio
async def test_renderer_client_rejects_empty_dict_native_response():
    client = object.__new__(RendererClient)

    with pytest.raises(EmptyModelResponseError):
        await client.raise_from_native_response({})


class _BridgeRenderer:
    supports_tools = True

    def __init__(self, bridge_base=None, bridge_full=None):
        self.bridge_base = bridge_base or [10, 99, 30]
        self.bridge_full = bridge_full or [10, 99, 30, 40, 50]
        self.calls = []

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        self.calls.append((messages, tools, add_generation_prompt))
        if len(messages) == 1 and add_generation_prompt is False:
            return list(self.bridge_base)
        if len(messages) > 1 and add_generation_prompt is True:
            return list(self.bridge_full)
        raise AssertionError((messages, tools, add_generation_prompt))

    def parse_response(self, token_ids):
        return ParsedResponse(content="")

    def get_stop_token_ids(self):
        return [99]


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ([{"role": "tool", "content": "a"}], True),
        ([{"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}], True),
        ([{"role": "user", "content": "next"}], True),
        ([{"role": "tool", "content": "a"}, {"role": "user", "content": "next"}], True),
        ([{"role": "assistant", "content": "no"}], False),
        (
            [{"role": "user", "content": "next"}, {"role": "tool", "content": "late"}],
            False,
        ),
    ],
)
def test_incremental_tail_accepts_tool_and_user_followups(tail, expected):
    assert _is_valid_incremental_tail(tail) is expected


@pytest.mark.asyncio
async def test_get_incremental_prompt_ids_matches_tool_tail_without_rerendering_completion():
    renderer = _BridgeRenderer(bridge_base=[10, 99, 30], bridge_full=[10, 99, 30, 40])
    prompt_messages = [SystemMessage(content="s"), UserMessage(content="u")]
    completion_messages = [
        AssistantMessage(
            content=None,
            tool_calls=[ToolCall(id="call_0", name="lookup", arguments="{}")],
        )
    ]
    prompt = [
        *[_to_renderer_message(m) for m in prompt_messages + completion_messages],
        _to_renderer_message(ToolMessage(content="result", tool_call_id="call_0")),
    ]
    state = {
        "trajectory": [
            {
                "prompt": prompt_messages,
                "completion": completion_messages,
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 99],
                    "is_truncated": False,
                },
                "is_truncated": False,
            }
        ]
    }

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=prompt, state=state, tools=None
    )

    assert result == [1, 2, 3, 99, 30, 40]
    assert len(renderer.calls) == 2


@pytest.mark.asyncio
async def test_get_incremental_prompt_ids_accepts_tool_then_user_tail():
    renderer = _BridgeRenderer(bridge_base=[10, 99], bridge_full=[10, 99, 40, 50])
    prompt_messages = [SystemMessage(content="s"), UserMessage(content="u")]
    completion_messages = [
        AssistantMessage(
            content=None,
            tool_calls=[ToolCall(id="call_0", name="lookup", arguments="{}")],
        )
    ]
    prompt = [
        *[_to_renderer_message(m) for m in prompt_messages + completion_messages],
        _to_renderer_message(ToolMessage(content="result", tool_call_id="call_0")),
        _to_renderer_message(UserMessage(content="continue")),
    ]
    state = {
        "trajectory": [
            {
                "prompt": prompt_messages,
                "completion": completion_messages,
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 99],
                    "is_truncated": False,
                },
                "is_truncated": False,
            }
        ]
    }

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=prompt, state=state, tools=None
    )

    assert result == [1, 2, 3, 99, 40, 50]


@pytest.mark.asyncio
async def test_get_incremental_prompt_ids_accepts_multimodal_tool_user_tail():
    renderer = _BridgeRenderer(bridge_base=[10, 99], bridge_full=[10, 99, 40, 50])
    prompt_messages = [
        SystemMessage(content="s"),
        UserMessage(
            content=[
                {"type": "text", "text": "inspect"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ]
        ),
    ]
    completion_messages = [
        AssistantMessage(
            content=None,
            tool_calls=[ToolCall(id="call_0", name="lookup", arguments="{}")],
        )
    ]
    prompt = [
        *[_to_renderer_message(m) for m in prompt_messages + completion_messages],
        _to_renderer_message(
            ToolMessage(
                content=[
                    {"type": "text", "text": "result"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,def"},
                    },
                ],
                tool_call_id="call_0",
            )
        ),
        _to_renderer_message(UserMessage(content="continue")),
    ]
    state = {
        "trajectory": [
            {
                "prompt": prompt_messages,
                "completion": completion_messages,
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 99],
                    "is_truncated": False,
                },
                "is_truncated": False,
            }
        ]
    }

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=prompt, state=state, tools=None
    )

    assert result == [1, 2, 3, 99, 40, 50]
