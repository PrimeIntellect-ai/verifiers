"""Tests for trajectory-based processing.

Covers:
- parse_response_tokens for extracting tokens from vf.Response
- Trajectory step processing for training data
- Handling of missing token data
"""

from unittest.mock import MagicMock

import pytest

from verifiers.types import (
    Response,
    ResponseMessage,
    ResponseTokens,
    State,
    TrajectoryStep,
    TrajectoryStepTokens,
)
from verifiers.utils.response_utils import parse_response_tokens


@pytest.mark.asyncio
async def test_parse_response_tokens_with_tokens():
    """Test parsing tokens from vf.Response with token data."""
    response = Response(
        id="test-id",
        created=0,
        model="test-model",
        message=ResponseMessage(
            role="assistant",
            content="Hello",
            reasoning_content=None,
            tool_calls=None,
            finish_reason="stop",
            is_truncated=False,
            tokens=ResponseTokens(
                prompt_ids=[1, 2, 3],
                prompt_mask=[0, 0, 0],
                completion_ids=[4, 5, 6],
                completion_mask=[1, 1, 1],
                completion_logprobs=[-0.1, -0.2, -0.3],
            ),
        ),
    )

    tokens = await parse_response_tokens(response)

    assert tokens is not None
    assert tokens["prompt_ids"] == [1, 2, 3]
    assert tokens["completion_ids"] == [4, 5, 6]
    assert tokens["prompt_mask"] == [0, 0, 0]
    assert tokens["completion_mask"] == [1, 1, 1]
    assert tokens["completion_logprobs"] == [-0.1, -0.2, -0.3]


@pytest.mark.asyncio
async def test_parse_response_tokens_without_tokens():
    """Test parsing tokens from vf.Response without token data."""
    response = Response(
        id="test-id",
        created=0,
        model="test-model",
        message=ResponseMessage(
            role="assistant",
            content="Hello",
            reasoning_content=None,
            tool_calls=None,
            finish_reason="stop",
            is_truncated=False,
            tokens=None,
        ),
    )

    tokens = await parse_response_tokens(response)

    assert tokens is None


@pytest.mark.asyncio
async def test_parse_response_tokens_with_max_seq_len_truncates_completion():
    """Test max_seq_len truncation for completion tokens."""
    response = Response(
        id="test-id",
        created=0,
        model="test-model",
        message=ResponseMessage(
            role="assistant",
            content="Hello",
            reasoning_content=None,
            tool_calls=None,
            finish_reason="length",
            is_truncated=True,
            tokens=ResponseTokens(
                prompt_ids=[10, 20],
                prompt_mask=[0, 0],
                completion_ids=[30, 40, 50],
                completion_mask=[1, 1, 1],
                completion_logprobs=[-0.5, -0.6, -0.7],
            ),
        ),
    )

    tokens = await parse_response_tokens(response, max_seq_len=4)

    assert tokens is not None
    assert tokens["prompt_ids"] == [10, 20]
    assert tokens["completion_ids"] == [30, 40]
    assert tokens["prompt_mask"] == [0, 0]
    assert tokens["completion_mask"] == [1, 1]
    assert tokens["completion_logprobs"] == [-0.5, -0.6]
    assert tokens["is_truncated"] is True


@pytest.mark.asyncio
async def test_parse_response_tokens_with_overlong_prompt():
    """Test overlong prompt handling with max_seq_len."""
    response = Response(
        id="test-id",
        created=0,
        model="test-model",
        message=ResponseMessage(
            role="assistant",
            content="Hello",
            reasoning_content=None,
            tool_calls=None,
            finish_reason="length",
            is_truncated=True,
            tokens=ResponseTokens(
                prompt_ids=[1, 2, 3, 4],
                prompt_mask=[0, 0, 0, 0],
                completion_ids=[5, 6],
                completion_mask=[1, 1],
                completion_logprobs=[-0.1, -0.2],
            ),
        ),
    )

    tokens = await parse_response_tokens(response, max_seq_len=3)

    assert tokens is not None
    assert tokens["prompt_ids"] == [1, 2, 3]
    assert tokens["completion_ids"] == []
    assert tokens["overlong_prompt"] is True
    assert tokens["is_truncated"] is True


@pytest.mark.asyncio
async def test_parse_response_tokens_carries_prompt_attribution():
    """``prompt_attribution`` on ResponseTokens migrates to the parsed
    TrajectoryStepTokens, and the response-side reference is cleared so
    downstream serialisation passes don't dedupe a duplicate (matches the
    move-not-copy policy used by ``multi_modal_data``).
    """
    from renderers.base import RenderedTokens

    attribution = RenderedTokens(
        token_ids=[1, 2, 3, 4],
        message_indices=[0, 0, 1, 1],
        sampled_mask=[False, False, False, False],
        is_content=[False, True, False, True],
        message_roles=["user", "tool"],
    )
    tokens_in = ResponseTokens(
        prompt_ids=[1, 2, 3, 4],
        prompt_mask=[0, 0, 0, 0],
        completion_ids=[5, 6],
        completion_mask=[1, 1],
        completion_logprobs=[-0.1, -0.2],
        prompt_attribution=attribution,
    )
    response = Response(
        id="test-id",
        created=0,
        model="test-model",
        message=ResponseMessage(
            role="assistant",
            content="Hello",
            reasoning_content=None,
            tool_calls=None,
            finish_reason="stop",
            is_truncated=False,
            tokens=tokens_in,
        ),
    )

    out = await parse_response_tokens(response)

    assert out is not None
    assert "prompt_attribution" in out
    out_attr = out["prompt_attribution"]
    # Same object — no copy, the sidecar moves from response.tokens to
    # the parsed step, matching the ``multi_modal_data`` policy.
    assert out_attr is attribution
    # And the response-side reference is cleared so save/msgpack passes
    # don't end up serialising the attribution twice.
    assert tokens_in.prompt_attribution is None


@pytest.mark.asyncio
async def test_parse_response_tokens_truncates_prompt_attribution_with_overlong_prompt():
    """When the prompt is overlong and gets truncated to ``max_seq_len``,
    the per-token arrays inside ``prompt_attribution`` (``token_ids``,
    ``message_indices``, ``sampled_mask``, ``is_content``) are sliced
    in lockstep with ``prompt_ids``. ``message_roles`` stays intact
    because it's indexed by *message position*, not token position —
    surviving ``message_indices`` values still resolve to the right
    role even after truncation drops some trailing messages."""
    from renderers.base import RenderedTokens

    attribution = RenderedTokens(
        token_ids=[10, 11, 12, 13, 14],
        message_indices=[0, 0, 1, 1, 1],
        sampled_mask=[False, False, False, False, False],
        is_content=[False, True, False, True, True],
        message_roles=["user", "tool"],
    )
    response = Response(
        id="test-id",
        created=0,
        model="test-model",
        message=ResponseMessage(
            role="assistant",
            content="Hello",
            reasoning_content=None,
            tool_calls=None,
            finish_reason="length",
            is_truncated=True,
            tokens=ResponseTokens(
                prompt_ids=[10, 11, 12, 13, 14],
                prompt_mask=[0, 0, 0, 0, 0],
                completion_ids=[20, 21],
                completion_mask=[1, 1],
                completion_logprobs=[-0.1, -0.2],
                prompt_attribution=attribution,
            ),
        ),
    )

    tokens = await parse_response_tokens(response, max_seq_len=3)

    assert tokens is not None
    assert tokens["overlong_prompt"] is True
    out_attr = tokens["prompt_attribution"]
    assert out_attr.token_ids == [10, 11, 12]
    assert out_attr.message_indices == [0, 0, 1]
    assert out_attr.sampled_mask == [False, False, False]
    assert out_attr.is_content == [False, True, False]
    # message_roles is per-message metadata, not per-token; both roles
    # still appear because message_indices[2] == 1 surviving the slice
    # still needs message_roles[1] to resolve correctly.
    assert out_attr.message_roles == ["user", "tool"]


@pytest.mark.asyncio
async def test_parse_response_tokens_leaves_prompt_attribution_when_only_completion_truncated():
    """When the prompt fits under ``max_seq_len`` and only the completion
    overflows, ``prompt_attribution`` is unaffected — it only describes
    the prompt portion."""
    from renderers.base import RenderedTokens

    attribution = RenderedTokens(
        token_ids=[1, 2],
        message_indices=[0, 0],
        sampled_mask=[False, False],
        is_content=[False, True],
        message_roles=["user"],
    )
    response = Response(
        id="test-id",
        created=0,
        model="test-model",
        message=ResponseMessage(
            role="assistant",
            content="Hello",
            reasoning_content=None,
            tool_calls=None,
            finish_reason="length",
            is_truncated=True,
            tokens=ResponseTokens(
                prompt_ids=[1, 2],
                prompt_mask=[0, 0],
                completion_ids=[3, 4, 5],
                completion_mask=[1, 1, 1],
                completion_logprobs=[-0.1, -0.2, -0.3],
                prompt_attribution=attribution,
            ),
        ),
    )

    tokens = await parse_response_tokens(response, max_seq_len=4)

    assert tokens is not None
    # Completion was truncated to fit within the budget.
    assert tokens["completion_ids"] == [3, 4]
    out_attr = tokens["prompt_attribution"]
    # Prompt-side attribution unchanged — identity-equal to the input.
    assert out_attr is attribution
    assert out_attr.token_ids == [1, 2]
    assert out_attr.is_content == [False, True]


# ---------------------------------------------------------------------------
# derive_prompt_message_tool_names — per-message tool function name lookup.
# ---------------------------------------------------------------------------


def test_derive_prompt_message_tool_names_returns_none_without_attribution():
    """Non-renderer client rollouts carry no ``prompt_attribution``;
    the helper short-circuits to ``None`` so callers can omit the
    field on the trajectory step without branching."""
    from verifiers.utils.response_utils import derive_prompt_message_tool_names

    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    assert derive_prompt_message_tool_names(msgs, None) is None


def test_derive_prompt_message_tool_names_returns_empty_list_on_empty_attribution():
    """Empty ``message_roles`` (renderer covered zero messages) →
    empty list, distinct from ``None`` (no attribution at all)."""
    from renderers.base import RenderedTokens

    from verifiers.utils.response_utils import derive_prompt_message_tool_names

    attribution = RenderedTokens(
        token_ids=[],
        message_indices=[],
        sampled_mask=[],
        is_content=[],
        message_roles=[],
    )
    out = derive_prompt_message_tool_names([], attribution)
    assert out == []


def test_derive_prompt_message_tool_names_first_turn_render():
    """On the first turn ``prompt_attribution.message_roles`` covers
    the full conversation. The helper produces one entry per message:
    the tool name for tool messages whose ``tool_call_id`` resolves
    to a preceding assistant's ``tool_calls``, ``None`` otherwise.
    """
    from renderers.base import RenderedTokens

    from verifiers.utils.response_utils import derive_prompt_message_tool_names

    msgs = [
        {"role": "user", "content": "What's 6*7?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "c1",
                    "function": {"name": "calc", "arguments": '{"e": "6*7"}'},
                }
            ],
        },
        {"role": "tool", "content": "42", "tool_call_id": "c1"},
        {"role": "user", "content": "What's the capital of France?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "c2",
                    "function": {"name": "lookup", "arguments": '{"q": "capital"}'},
                }
            ],
        },
        {"role": "tool", "content": "Paris", "tool_call_id": "c2"},
    ]
    attribution = RenderedTokens(
        token_ids=[0] * 6,  # don't care about actual tokens here
        message_indices=[],
        sampled_mask=[],
        is_content=[],
        message_roles=[m["role"] for m in msgs],
    )
    out = derive_prompt_message_tool_names(msgs, attribution)
    assert out == [None, None, "calc", None, None, "lookup"]


def test_derive_prompt_message_tool_names_bridge_with_resolvable_caller():
    """In the bridge path ``message_roles`` covers only ``new_messages``
    (the tail of the prompt). When a tool message's issuing assistant
    is *also* in that tail (a complete tool cycle within the new
    portion), its name resolves from the in-slice lookup."""
    from renderers.base import RenderedTokens

    from verifiers.utils.response_utils import derive_prompt_message_tool_names

    full_prompt = [
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "First reply"},
        # The bridge starts here — new_messages is the tail.
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "c1",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
    ]
    # Bridge attribution covers the trailing 2 messages.
    attribution = RenderedTokens(
        token_ids=[],
        message_indices=[],
        sampled_mask=[],
        is_content=[],
        message_roles=["assistant", "tool"],
    )
    out = derive_prompt_message_tool_names(full_prompt, attribution)
    assert out == [None, "lookup"]


def test_derive_prompt_message_tool_names_bridge_with_orphan_tool_message():
    """Tool message whose issuing assistant is in the prior portion of
    a bridged turn is an *orphan* — the bridge attribution only covers
    the tail, so the lookup misses and the entry is ``None``. This is
    the documented edge case: trainers should treat orphans as
    "untrainable" via the natural ``None`` fallthrough."""
    from renderers.base import RenderedTokens

    from verifiers.utils.response_utils import derive_prompt_message_tool_names

    full_prompt = [
        # The assistant that issued the tool_call lives in the prior
        # portion — outside the bridge's covered slice.
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "c1",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        # Bridge covers from here on: just the tool message.
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
    ]
    attribution = RenderedTokens(
        token_ids=[],
        message_indices=[],
        sampled_mask=[],
        is_content=[],
        message_roles=["tool"],
    )
    out = derive_prompt_message_tool_names(full_prompt, attribution)
    # Orphan — issuing assistant not in the covered slice → None.
    assert out == [None]


def test_derive_prompt_message_tool_names_accepts_pydantic_messages():
    """``MultiTurnEnv`` passes prompts through that may be either
    Pydantic ``Message`` instances (``AssistantMessage``,
    ``ToolMessage``) or plain dicts depending on the env. The helper
    must handle both — uses ``getattr`` then falls back to dict
    access."""
    from renderers.base import RenderedTokens

    from verifiers.types import AssistantMessage, ToolCall, ToolMessage, UserMessage
    from verifiers.utils.response_utils import derive_prompt_message_tool_names

    msgs = [
        UserMessage(content="hi"),
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id="c1", name="lookup", arguments="{}"),
            ],
        ),
        ToolMessage(content="result", tool_call_id="c1"),
    ]
    attribution = RenderedTokens(
        token_ids=[],
        message_indices=[],
        sampled_mask=[],
        is_content=[],
        message_roles=["user", "assistant", "tool"],
    )
    out = derive_prompt_message_tool_names(msgs, attribution)
    assert out == [None, None, "lookup"]


def test_derive_prompt_message_tool_names_handles_tool_call_without_function_envelope():
    """``ToolCall`` Pydantic exposes ``name`` directly; OpenAI dict
    shape nests it under ``function.name``. The helper accepts both —
    no caller has to normalise before calling it."""
    from renderers.base import RenderedTokens

    from verifiers.utils.response_utils import derive_prompt_message_tool_names

    msgs = [
        # ToolCall in flat Pydantic shape (no "function" key)
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "name": "lookup", "arguments": "{}"}],
        },
        {"role": "tool", "content": "ok", "tool_call_id": "c1"},
    ]
    attribution = RenderedTokens(
        token_ids=[],
        message_indices=[],
        sampled_mask=[],
        is_content=[],
        message_roles=["assistant", "tool"],
    )
    out = derive_prompt_message_tool_names(msgs, attribution)
    assert out == [None, "lookup"]


def test_process_trajectory_steps_for_training(make_input):
    """Test processing trajectory steps into training examples."""
    state1 = State(
        input=make_input(
            prompt=[{"role": "user", "content": "Hello"}],
        )
    )
    state1["trajectory"] = [
        TrajectoryStep(
            prompt=[{"role": "user", "content": "Hello"}],
            completion=[{"role": "assistant", "content": "Hi"}],
            response=MagicMock(),
            tokens=TrajectoryStepTokens(
                prompt_ids=[1, 2],
                prompt_mask=[0, 0],
                completion_ids=[3, 4],
                completion_mask=[1, 1],
                completion_logprobs=[-0.1, -0.2],
                overlong_prompt=False,
                is_truncated=False,
            ),
            reward=1.0,
            advantage=None,
            is_truncated=False,
            trajectory_id="test_trajectory",
            extras={},
        )
    ]

    state2 = State(
        input=make_input(
            prompt=[{"role": "user", "content": "Bye"}],
            example_id=1,
        )
    )
    state2["trajectory"] = [
        TrajectoryStep(
            prompt=[{"role": "user", "content": "Bye"}],
            completion=[{"role": "assistant", "content": "Goodbye"}],
            response=MagicMock(),
            tokens=TrajectoryStepTokens(
                prompt_ids=[5],
                prompt_mask=[0],
                completion_ids=[6, 7, 8],
                completion_mask=[1, 1, 1],
                completion_logprobs=[-0.3, -0.4, -0.5],
                overlong_prompt=False,
                is_truncated=False,
            ),
            reward=0.5,
            advantage=None,
            is_truncated=False,
            trajectory_id="test_trajectory",
            extras={},
        )
    ]

    states = [state1, state2]

    # Process trajectories horizontally - each step becomes a separate training example
    prompt_ids_list = []
    completion_ids_list = []
    completion_logprobs_list = []
    prompt_mask_list = []
    completion_mask_list = []
    rewards_list = []

    for state in states:
        trajectory = state["trajectory"]
        for step in trajectory:
            if step["tokens"] is None:
                continue
            tokens = step["tokens"]
            prompt_ids_list.append(tokens["prompt_ids"])
            completion_ids_list.append(tokens["completion_ids"])
            completion_logprobs_list.append(tokens["completion_logprobs"])
            prompt_mask_list.append(tokens["prompt_mask"])
            completion_mask_list.append(tokens["completion_mask"])
            rewards_list.append(step.get("reward", 0.0))

    assert len(prompt_ids_list) == 2
    assert prompt_ids_list[0] == [1, 2]
    assert prompt_ids_list[1] == [5]
    assert completion_ids_list[0] == [3, 4]
    assert completion_ids_list[1] == [6, 7, 8]
    assert completion_logprobs_list[0] == [-0.1, -0.2]
    assert completion_logprobs_list[1] == [-0.3, -0.4, -0.5]
    assert rewards_list == [1.0, 0.5]


def test_process_trajectory_steps_skip_missing_tokens(make_input):
    """Test that trajectory steps without tokens are skipped."""
    state = State(
        input=make_input(
            prompt=[{"role": "user", "content": "Hello"}],
        )
    )
    state["trajectory"] = [
        TrajectoryStep(
            prompt=[{"role": "user", "content": "Hello"}],
            completion=[{"role": "assistant", "content": "Hi"}],
            response=MagicMock(),
            tokens=None,
            reward=1.0,
            advantage=None,
            is_truncated=False,
            trajectory_id="test_trajectory",
            extras={},
        ),
        TrajectoryStep(
            prompt=[{"role": "user", "content": "Hello"}],
            completion=[{"role": "assistant", "content": "Hi again"}],
            response=MagicMock(),
            tokens=TrajectoryStepTokens(
                prompt_ids=[1],
                prompt_mask=[0],
                completion_ids=[2, 3],
                completion_mask=[1, 1],
                completion_logprobs=[-0.1, -0.2],
                overlong_prompt=False,
                is_truncated=False,
            ),
            reward=0.5,
            advantage=None,
            is_truncated=False,
            trajectory_id="test_trajectory",
            extras={},
        ),
    ]

    processed_steps = []
    for step in state["trajectory"]:
        if step["tokens"] is not None:
            processed_steps.append(step)

    assert len(processed_steps) == 1
    assert processed_steps[0]["tokens"] is not None
    assert processed_steps[0]["reward"] == 0.5


def test_trajectory_step_mask_combining():
    """Test combining prompt and completion masks for training."""
    tokens = TrajectoryStepTokens(
        prompt_ids=[1, 2, 3],
        prompt_mask=[0, 0, 0],
        completion_ids=[4, 5],
        completion_mask=[1, 1],
        completion_logprobs=[-0.1, -0.2],
    )

    # Combine for training
    token_ids = tokens["prompt_ids"] + tokens["completion_ids"]
    mask = tokens["prompt_mask"] + tokens["completion_mask"]
    logprobs = [0.0] * len(tokens["prompt_ids"]) + tokens["completion_logprobs"]

    assert token_ids == [1, 2, 3, 4, 5]
    assert mask == [0, 0, 0, 1, 1]
    assert logprobs == [0.0, 0.0, 0.0, -0.1, -0.2]
