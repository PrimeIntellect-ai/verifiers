from functools import lru_cache
from unittest.mock import patch

import pytest

import verifiers as vf
from verifiers.errors import EmptyModelResponseError
from renderers import RendererPool
from renderers.base import ParsedResponse, create_renderer
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


# ── Parity across real renderers: truncated most-recent step ──────────
#
# When vLLM hits max_tokens mid-completion, the previous step carries
# is_truncated=True and completion_ids without an end-of-turn stop token.
# The anchor loop in _get_incremental_prompt_ids used to skip every
# truncated step regardless of whether the renderer opts in to
# synthesize-close, so the bridge never ran and the caller fell back to a
# full re-render. The extension property then broke whenever BPE
# round-trip diverged and the rollout fragmented.
#
# These tests run across every renderer in the parity matrix to make
# sure that regression stays fixed: with synth_ok, the bridge anchors on
# the truncated step and returns prefix-preserving ids; without synth_ok
# (DefaultRenderer default), it bails to None and the caller falls back.

# Mirror of packages/renderers/tests/conftest.py::RENDERER_MODELS so the
# bridge-over-truncation parity lines up with the render_ids parity.
#
# Some entries carry an xfail reason: those renderers have pre-existing
# bridge limitations independent of synthesize-close. The test still
# runs across them to document the current state and to auto-flip to a
# pass if the underlying renderer is fixed.
_TRUNCATED_ANCHOR_MODELS = [
    pytest.param("Qwen/Qwen3-8B", "auto", id="Qwen/Qwen3-8B"),
    pytest.param("Qwen/Qwen3.5-9B", "auto", id="Qwen/Qwen3.5-9B"),
    pytest.param(
        "zai-org/GLM-5",
        "auto",
        id="zai-org/GLM-5",
        marks=pytest.mark.xfail(
            reason="GLM family emits no per-turn close token; render_ids("
            "[dummy_assistant]) ends on raw content so "
            "build_incremental_prompt_ids can't find a boundary in "
            "bridge_base_ids. Pre-existing; not introduced by the anchor fix.",
            strict=False,
        ),
    ),
    pytest.param(
        "zai-org/GLM-4.7-Flash",
        "auto",
        id="zai-org/GLM-4.7-Flash",
        marks=pytest.mark.xfail(
            reason="Same GLM next-turn-marker template as GLM-5.",
            strict=False,
        ),
    ),
    pytest.param(
        "THUDM/GLM-4.5-Air",
        "auto",
        id="THUDM/GLM-4.5-Air",
        marks=pytest.mark.xfail(
            reason="Same GLM next-turn-marker template as GLM-5.",
            strict=False,
        ),
    ),
    pytest.param("MiniMaxAI/MiniMax-M2.5", "auto", id="MiniMaxAI/MiniMax-M2.5"),
    pytest.param(
        "Qwen/Qwen2.5-0.5B-Instruct", "default", id="Qwen/Qwen2.5-0.5B-Instruct"
    ),
]


@lru_cache(maxsize=None)
def _load_tokenizer_and_renderer(
    model_name: str, renderer_name: str, synth_close: bool
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs = {}
    # Only DefaultRenderer consumes synthesize_close_on_truncation; other
    # renderers hard-code it at the class level.
    if renderer_name == "default":
        kwargs["synthesize_close_on_truncation"] = synth_close
    renderer = create_renderer(tokenizer, renderer=renderer_name, **kwargs)
    return tokenizer, renderer


def _build_truncated_state(tokenizer, renderer):
    """Construct a single-step trajectory whose most-recent step is
    truncated. prev_prompt_ids / prev_completion_ids use the renderer's
    own tokens so the assertion reflects what the real orchestrator
    hands the bridge — the exact tokens vLLM produced for the partial
    assistant turn, with no end-of-turn marker at the end.
    """
    step_prompt = [{"role": "user", "content": "Guess a 5-letter word."}]
    prev_prompt_ids = renderer.render_ids(step_prompt, add_generation_prompt=True)
    truncated_text = (
        "I'll start with a common word. Let me think about this — "
        "the most frequent letters are E, A, R, I, O, T, N, S, L"
    )
    prev_completion_ids = tokenizer.encode(truncated_text, add_special_tokens=False)

    step_completion = [{"role": "assistant", "content": truncated_text}]
    state = {
        "trajectory": [
            {
                "prompt": step_prompt,
                "completion": step_completion,
                "tokens": {
                    "prompt_ids": list(prev_prompt_ids),
                    "completion_ids": list(prev_completion_ids),
                    "is_truncated": True,
                },
                "is_truncated": True,
            }
        ]
    }
    next_turn_prompt = step_prompt + step_completion + [
        {"role": "user", "content": "Your guess was invalid. Give a 5-letter word."}
    ]
    return prev_prompt_ids, prev_completion_ids, state, next_turn_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,renderer_name",
    _TRUNCATED_ANCHOR_MODELS,
)
async def test_get_incremental_prompt_ids_bridges_over_truncated_step(
    model_name, renderer_name
):
    """With synth_ok=True, the bridge anchors on the truncated step and
    returns new prompt_ids that start with prev_prompt + prev_completion
    byte-identically (the extension invariant). This is what keeps
    interleave_rollout from fragmenting the rollout into two samples."""
    tokenizer, renderer = _load_tokenizer_and_renderer(
        model_name, renderer_name, synth_close=True
    )
    prev_prompt_ids, prev_completion_ids, state, next_turn_prompt = (
        _build_truncated_state(tokenizer, renderer)
    )

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=next_turn_prompt, state=state, tools=None
    )

    prefix = list(prev_prompt_ids) + list(prev_completion_ids)
    assert result is not None, f"{model_name}: bridge returned None on truncated anchor"
    assert result[: len(prefix)] == prefix, (
        f"{model_name}: bridge result does not prefix-preserve "
        f"prev_prompt + prev_completion"
    )
    assert len(result) > len(prefix), (
        f"{model_name}: bridge produced no tail tokens for the new user turn"
    )


@pytest.mark.asyncio
async def test_get_incremental_prompt_ids_bails_for_default_renderer_without_synth_close():
    """Without synth_ok, DefaultRenderer must bail to None so the caller
    falls back to a full apply_chat_template re-render — preserving main's
    TITO-on-truncation behavior for anyone who hasn't opted in."""
    tokenizer, renderer = _load_tokenizer_and_renderer(
        "Qwen/Qwen2.5-0.5B-Instruct", "default", synth_close=False
    )
    _, _, state, next_turn_prompt = _build_truncated_state(tokenizer, renderer)

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=next_turn_prompt, state=state, tools=None
    )

    assert result is None
