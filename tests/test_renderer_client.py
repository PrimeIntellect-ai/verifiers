from functools import lru_cache
from unittest.mock import patch

import pytest

import verifiers as vf
from renderers import RendererPool
from renderers.base import ParsedResponse, RenderedTokens, create_renderer
from verifiers.clients.renderer_client import (
    RendererClient,
    _attach_tool_call_names,
    _get_incremental_prompt_ids,
    _is_valid_incremental_tail,
    _step_token_ids,
    _to_renderer_message,
)
from verifiers.errors import EmptyModelResponseError
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

    sentinel_pool = RendererPool.__new__(RendererPool)
    with patch(
        "verifiers.clients.renderer_client.create_renderer_pool",
        return_value=sentinel_pool,
    ) as create_pool_mock:
        pool = client._get_renderer_or_pool("Qwen/Qwen3-VL-4B-Instruct")

    assert pool is sentinel_pool
    create_pool_mock.assert_called_once_with(
        "Qwen/Qwen3-VL-4B-Instruct",
        renderer="qwen3_vl",
        size=1,
        tool_parser=None,
        reasoning_parser=None,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=False,
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

    sentinel_pool = RendererPool.__new__(RendererPool)
    with patch(
        "verifiers.clients.renderer_client.create_renderer_pool",
        return_value=sentinel_pool,
    ) as create_pool_mock:
        pool = client._get_renderer_or_pool("r8-smoke")

    assert pool is sentinel_pool
    create_pool_mock.assert_called_once_with(
        "Qwen/Qwen3-VL-4B-Instruct",
        renderer="qwen3_vl",
        size=1,
        tool_parser=None,
        reasoning_parser=None,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=False,
    )


# Provenance: Eli's review on PR #1068, comment 3150580768.
#   "RendererClient parses the GPT-OSS assistant tool call into ToolCall(name=...),
#   but ToolEnv returns ToolMessage with only content/tool_call_id, and
#   _to_renderer_message forwards only role/content/tool_call_id. As a result every
#   GPT-OSS tool result rendered through this path becomes `functions.unknown
#   to=assistant`."
# The verifiers ToolMessage schema has no `name` field, so per-message conversion
# can't recover it — we have to walk the list and look up the prior assistant
# tool_call by tool_call_id. This test pins the contract of that lookup.
def test_attach_tool_call_names_recovers_function_name_from_prior_call():
    messages = [
        _to_renderer_message(UserMessage(content="2+2?")),
        _to_renderer_message(
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(id="c1", name="calculator", arguments="{}"),
                ],
            )
        ),
        _to_renderer_message(ToolMessage(content="4", tool_call_id="c1")),
    ]
    out = _attach_tool_call_names(messages)
    assert out[2]["role"] == "tool"
    assert out[2]["name"] == "calculator"


def test_attach_tool_call_names_preserves_existing_name():
    """Caller-supplied `name` (already on a Mapping input) wins over recovery."""
    messages: list = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "calculator", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "name": "explicit",
            "content": "4",
            "tool_call_id": "c1",
        },
    ]
    out = _attach_tool_call_names(messages)
    assert out[1]["name"] == "explicit"


def test_attach_tool_call_names_unknown_tool_call_id_left_unset():
    """A tool message with no matching prior call gets no `name` (renderer
    falls back to its own default, e.g. GPT-OSS uses 'unknown')."""
    messages = [
        _to_renderer_message(UserMessage(content="hi")),
        _to_renderer_message(ToolMessage(content="orphan", tool_call_id="nope")),
    ]
    out = _attach_tool_call_names(messages)
    assert "name" not in out[1]


@pytest.mark.asyncio
async def test_to_native_tool_returns_openai_envelope():
    """``RendererClient.to_native_tool`` must wrap each ``Tool`` in the
    OpenAI envelope (``{"type": "function", "function": {...}}``) — the
    same shape ``OpenAIChatCompletionsClient`` sends server-side under
    TITO/MITO. Modern function-calling models (Qwen3 family, GLM, Kimi)
    saw the envelope at training time, so the renderer client's prompt
    must match. Regression for the bare-form bug where rollout-mode
    tool envs produced uniformly zero rewards because the model never
    emitted ``<tool_call>`` blocks under an out-of-distribution prompt.
    """
    from verifiers.types import Tool

    client = object.__new__(RendererClient)
    tool = Tool(
        name="get_weather",
        description="Get the weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )

    native = await client.to_native_tool(tool)
    assert native["type"] == "function"
    assert native["function"]["name"] == "get_weather"
    assert native["function"]["description"] == "Get the weather for a city"
    assert native["function"]["parameters"]["required"] == ["city"]
    assert "strict" not in native["function"]


@pytest.mark.asyncio
async def test_to_native_tool_propagates_strict_flag():
    """When ``Tool.strict`` is set the envelope must carry it through —
    OpenAI's strict-schema enforcement only kicks in on the inner function
    object, never the envelope itself."""
    from verifiers.types import Tool

    client = object.__new__(RendererClient)
    tool = Tool(
        name="get_weather",
        description="Get the weather for a city",
        parameters={"type": "object", "properties": {}},
        strict=True,
    )

    native = await client.to_native_tool(tool)
    assert native["function"]["strict"] is True


@pytest.mark.asyncio
async def test_renderer_client_accepts_dict_native_response_with_content():
    client = object.__new__(RendererClient)

    await client.raise_from_native_response({"content": "done"})


@pytest.mark.asyncio
async def test_renderer_client_rejects_empty_dict_native_response():
    client = object.__new__(RendererClient)

    with pytest.raises(EmptyModelResponseError):
        await client.raise_from_native_response({})


@pytest.mark.asyncio
async def test_from_native_response_uses_request_id_and_token_lengths():
    """vLLM's /inference/v1/generate returns ``request_id`` (not ``id``) and
    no ``usage``/``model``/``created``. ``Response.id`` should pick up
    ``request_id``; usage is reconstructed from token-list lengths;
    model/created are unused metadata.
    """
    client = object.__new__(RendererClient)
    response_dict = {
        "request_id": "resp-42",
        "content": "ok",
        "reasoning_content": None,
        "tool_calls": None,
        "finish_reason": "stop",
        "prompt_ids": [1, 2, 3],
        "completion_ids": [4, 5],
        "completion_logprobs": [-0.1, -0.2],
        "routed_experts": None,
    }

    response = await client.from_native_response(response_dict)
    assert response.id == "resp-42"
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 5


class _BridgeRenderer:
    supports_tools = True

    # Token ID 99 plays the role of the stop / end-of-turn marker in the
    # token streams these tests construct.
    _STOP_TOKEN_ID = 99

    def __init__(self, bridge_base=None, bridge_full=None):
        self.bridge_base = bridge_base or [10, 99, 30]
        self.bridge_full = bridge_full or [10, 99, 30, 40, 50]
        self.calls = []
        self.bridge_calls = 0

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        self.calls.append((messages, tools, add_generation_prompt))
        if len(messages) == 1 and add_generation_prompt is False:
            return list(self.bridge_base)
        if len(messages) > 1 and add_generation_prompt is True:
            return list(self.bridge_full)
        raise AssertionError((messages, tools, add_generation_prompt))

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
    ):
        """Mimic the Renderer Protocol's bridge contract enough for the
        ``_get_incremental_prompt_ids`` tests to exercise the post-bridge
        plumbing.

        Returns ``prev_prompt + prev_completion + trailing + extension``,
        where ``trailing`` is whatever ``bridge_base`` emits AFTER the stop
        token (the "turn boundary" tokens our renderers emit between turns)
        and ``extension`` is the suffix ``bridge_full`` adds on top of
        ``bridge_base``.
        """
        self.bridge_calls += 1
        # Find the stop token in bridge_base and split into close + trailing.
        try:
            stop_idx = self.bridge_base.index(self._STOP_TOKEN_ID)
        except ValueError:
            stop_idx = len(self.bridge_base) - 1
        trailing = list(self.bridge_base[stop_idx + 1 :])
        extension = list(self.bridge_full[len(self.bridge_base) :])
        return RenderedTokens(
            token_ids=(
                list(previous_prompt_ids)
                + list(previous_completion_ids)
                + trailing
                + extension
            )
        )

    def parse_response(self, token_ids, *, tools=None):
        return ParsedResponse(content="")

    def get_stop_token_ids(self):
        return [self._STOP_TOKEN_ID]


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

    assert result is not None
    assert result.token_ids == [1, 2, 3, 99, 30, 40]
    # The bridge stitches over the completion without re-rendering it —
    # one bridge call, zero render_ids calls (older diff-based bridges
    # called render_ids twice).
    assert renderer.bridge_calls == 1
    assert renderer.calls == []


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

    assert result is not None
    assert result.token_ids == [1, 2, 3, 99, 40, 50]


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

    assert result is not None
    assert result.token_ids == [1, 2, 3, 99, 40, 50]


# ── Parity across real renderers: truncated most-recent step ──────────
#
# When vLLM hits max_tokens mid-completion, the previous step carries
# is_truncated=True and completion_ids without an end-of-turn stop token.
# The anchor loop in _get_incremental_prompt_ids used to skip every
# truncated step, so the bridge never ran and the caller fell back to a
# full re-render. The extension property then broke whenever BPE
# round-trip diverged and the rollout fragmented.
#
# These tests run across every hand-coded renderer in the parity matrix
# to make sure that regression stays fixed: the bridge anchors on the
# truncated step and returns prefix-preserving ids.

# Mirror of packages/renderers/tests/conftest.py::RENDERER_MODELS,
# restricted to hand-coded renderers (DefaultRenderer never bridges by
# design — covered separately in the bails-on-default test below).
_TRUNCATED_ANCHOR_MODELS = [
    pytest.param("Qwen/Qwen3-8B", "auto", id="Qwen/Qwen3-8B"),
    pytest.param("Qwen/Qwen3.5-9B", "auto", id="Qwen/Qwen3.5-9B"),
    pytest.param("Qwen/Qwen3-VL-4B-Instruct", "auto", id="Qwen/Qwen3-VL-4B-Instruct"),
    pytest.param("zai-org/GLM-5", "auto", id="zai-org/GLM-5"),
    pytest.param("zai-org/GLM-4.7-Flash", "auto", id="zai-org/GLM-4.7-Flash"),
    pytest.param("THUDM/GLM-4.5-Air", "auto", id="THUDM/GLM-4.5-Air"),
    pytest.param("MiniMaxAI/MiniMax-M2.5", "auto", id="MiniMaxAI/MiniMax-M2.5"),
    pytest.param(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "auto",
        id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ),
    pytest.param("openai/gpt-oss-20b", "gpt-oss", id="openai/gpt-oss-20b"),
]


@lru_cache(maxsize=None)
def _load_tokenizer_and_renderer(model_name: str, renderer_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    renderer = create_renderer(tokenizer, renderer=renderer_name)
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
    next_turn_prompt = (
        step_prompt
        + step_completion
        + [{"role": "user", "content": "Your guess was invalid. Give a 5-letter word."}]
    )
    return prev_prompt_ids, prev_completion_ids, state, next_turn_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_id,renderer_id",
    _TRUNCATED_ANCHOR_MODELS,
)
async def test_get_incremental_prompt_ids_bridges_over_truncated_step(
    model_id, renderer_id
):
    """The bridge anchors on the truncated step and returns new prompt_ids
    that start with prev_prompt + prev_completion byte-identically (the
    extension invariant). This is what keeps interleave_rollout from
    fragmenting the rollout into two samples."""
    tokenizer, renderer = _load_tokenizer_and_renderer(model_id, renderer_id)
    prev_prompt_ids, prev_completion_ids, state, next_turn_prompt = (
        _build_truncated_state(tokenizer, renderer)
    )

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=next_turn_prompt, state=state, tools=None
    )

    prefix = list(prev_prompt_ids) + list(prev_completion_ids)
    assert result is not None, f"{model_id}: bridge returned None on truncated anchor"
    result_ids = result.token_ids
    assert result_ids[: len(prefix)] == prefix, (
        f"{model_id}: bridge result does not prefix-preserve "
        f"prev_prompt + prev_completion"
    )
    assert len(result_ids) > len(prefix), (
        f"{model_id}: bridge produced no tail tokens for the new user turn"
    )


@pytest.mark.asyncio
async def test_get_incremental_prompt_ids_bails_for_default_renderer():
    """DefaultRenderer always bails to None so the caller falls back to a
    full apply_chat_template re-render — preserving main's
    TITO-on-truncation behavior for unknown templates."""
    tokenizer, renderer = _load_tokenizer_and_renderer(
        "Qwen/Qwen2.5-0.5B-Instruct", "default"
    )
    _, _, state, next_turn_prompt = _build_truncated_state(tokenizer, renderer)

    result = await _get_incremental_prompt_ids(
        renderer=renderer, prompt=next_turn_prompt, state=state, tools=None
    )

    assert result is None


# ── _step_token_ids: fallback to raw response tokens when step.tokens was
# truncated by parse_response_tokens (empty completion_ids / prompt_ids on
# overlong turns). The bridge needs un-truncated anchor tokens to chain
# across turns; training budget is still enforced at sample-assembly time.


class _RawTokens:
    def __init__(self, prompt_ids, completion_ids):
        self.prompt_ids = prompt_ids
        self.completion_ids = completion_ids


class _ResponseMessage:
    def __init__(self, tokens):
        self.tokens = tokens


class _Response:
    def __init__(self, message):
        self.message = message


def test_step_token_ids_happy_path():
    """Populated step.tokens → returns those directly without inspecting response."""
    step = {
        "tokens": {"prompt_ids": [1, 2, 3], "completion_ids": [4, 5]},
        "response": None,
    }
    assert _step_token_ids(step) == ([1, 2, 3], [4, 5])


def test_step_token_ids_falls_back_on_empty_completion():
    """Overlong prompt: parse_response_tokens zeros completion_ids in step.tokens
    but raw tokens on step.response.message.tokens are still intact. Fallback
    must return the raw tokens so bridge can extend past the overlong turn."""
    step = {
        "tokens": {"prompt_ids": [1, 2, 3], "completion_ids": []},
        "response": _Response(
            _ResponseMessage(_RawTokens([10, 11, 12, 13], [14, 15, 16]))
        ),
    }
    assert _step_token_ids(step) == ([10, 11, 12, 13], [14, 15, 16])


def test_step_token_ids_falls_back_when_tokens_is_none():
    """step.tokens==None (e.g. agent-completed sentinel or uninitialized step)
    should still fall back to raw response tokens when available."""
    step = {
        "tokens": None,
        "response": _Response(_ResponseMessage(_RawTokens([20, 21], [22, 23]))),
    }
    assert _step_token_ids(step) == ([20, 21], [22, 23])


def test_step_token_ids_returns_none_when_both_sources_empty():
    """If both step.tokens and response.message.tokens are empty/absent, must
    return None (caller falls back to full re-render)."""
    step = {
        "tokens": {"prompt_ids": [], "completion_ids": []},
        "response": _Response(_ResponseMessage(_RawTokens([], []))),
    }
    assert _step_token_ids(step) is None


def test_step_token_ids_returns_none_when_truncated_and_no_response():
    """Guard: empty step.tokens + no response object → None, not an AttributeError."""
    step = {
        "tokens": {"prompt_ids": [1, 2], "completion_ids": []},
        "response": None,
    }
    assert _step_token_ids(step) is None
