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
    """``prompt_attribution`` moves from ResponseTokens to the parsed step
    as a JSON-safe dict (move-not-copy, same policy as ``multi_modal_data``)."""
    import dataclasses

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
    assert out["prompt_attribution"] == dataclasses.asdict(attribution)
    assert tokens_in.prompt_attribution is None


@pytest.mark.asyncio
async def test_parse_response_tokens_truncates_prompt_attribution_with_overlong_prompt():
    """On overlong-prompt truncation, the per-token arrays inside
    ``prompt_attribution`` get sliced in lockstep with ``prompt_ids``;
    ``message_roles`` stays intact (it's per-message, not per-token)."""
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
    assert isinstance(out_attr, dict)
    assert out_attr["token_ids"] == [10, 11, 12]
    assert out_attr["message_indices"] == [0, 0, 1]
    assert out_attr["sampled_mask"] == [False, False, False]
    assert out_attr["is_content"] == [False, True, False]
    assert out_attr["message_roles"] == ["user", "tool"]


@pytest.mark.asyncio
async def test_parsed_prompt_attribution_survives_v1_assert_serializable():
    """Regression for PR #1414: a v1 state with ``prompt_attribution``
    on a trajectory step must clear ``State.assert_serializable`` (the
    ``json.dumps`` gate that pre-fix raised on the RenderedTokens dataclass).
    """
    from renderers.base import RenderedTokens

    from verifiers.v1.utils.serialization_utils import serializable

    response = Response(
        id="t",
        created=0,
        model="m",
        message=ResponseMessage(
            role="assistant",
            content="hi",
            reasoning_content=None,
            tool_calls=None,
            finish_reason="stop",
            is_truncated=False,
            tokens=ResponseTokens(
                prompt_ids=[1, 2],
                prompt_mask=[0, 0],
                completion_ids=[3],
                completion_mask=[1],
                completion_logprobs=[-0.1],
                prompt_attribution=RenderedTokens(
                    token_ids=[1, 2],
                    message_indices=[0, 0],
                    sampled_mask=[False, False],
                    is_content=[False, True],
                    message_roles=["user"],
                ),
            ),
        ),
    )
    parsed = await parse_response_tokens(response)
    step = {"tokens": serializable(parsed), "response": serializable(response)}
    State({"trajectory": [step]}).assert_serializable()


def test_assert_serializable_accepts_msgpack_sidecars_rejects_unknown():
    """The ``assert_serializable`` json.dumps gate must accept exactly what the
    trainer transport (msgpack) accepts, while staying strict otherwise.

    Trajectory token steps carry sidecars that are non-JSON by design and reach
    the trainer via msgpack, not JSON: the renderer ``MultiModalData`` (a
    dataclass holding numpy pixel arrays) and ``routed_experts`` (a raw
    ``memoryview`` buffer). Both must clear the gate; any other
    non-serializable object must still raise.
    """
    import dataclasses

    import numpy as np

    @dataclasses.dataclass
    class _FakeMultiModalData:
        mm_hashes: dict
        mm_items: dict
        mm_placeholders: dict

    mm = _FakeMultiModalData(
        mm_hashes={"image": ["h1"]},
        mm_items={"image": [np.zeros((2, 2), dtype=np.uint8)]},
        mm_placeholders={"image": [{"offset": 0, "length": 4}]},
    )
    step = {
        "tokens": {
            "prompt_ids": [1, 2],
            "multi_modal_data": mm,
            "routed_experts": {"data": memoryview(b"abc"), "shape": [3], "start": 0},
        }
    }
    # Must not raise: both sidecars are msgpack-transported, not JSON.
    State({"trajectory": [step]}).assert_serializable()

    # A genuinely non-serializable object must still be rejected.
    class _Unknown:
        pass

    with pytest.raises(TypeError):
        State({"trajectory": [{"tokens": _Unknown()}]}).assert_serializable()


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


def test_strip_routed_experts_data_key_order_robust():
    """The zero-copy stripper must find ``data`` regardless of key order
    (``dtype``/``shape``/``start`` may precede it) and no-op when absent."""
    from verifiers.utils.response_utils import strip_routed_experts_data

    # data first (fast path)
    raw = b'{"routed_experts":{"data":"QUJD","shape":[3],"start":0,"dtype":"uint8"}}'
    stripped, blob = strip_routed_experts_data(raw)
    assert blob is not None and blob.tobytes() == b"QUJD"
    assert b'"data":""' in stripped

    # dtype/shape/start before data — must still strip the blob
    raw2 = b'{"routed_experts":{"dtype":"uint16","shape":[3],"start":0,"data":"WFla"}}'
    stripped2, blob2 = strip_routed_experts_data(raw2)
    assert blob2 is not None and blob2.tobytes() == b"WFla"
    assert b'"data":""' in stripped2

    # routed_experts object lacks data; an unrelated sibling has data — must
    # NOT be mistaken for routed experts (search bounded to the object).
    raw4 = b'{"routed_experts":{"shape":[3],"start":0},"other":{"data":"UNRELATED"}}'
    stripped4, blob4 = strip_routed_experts_data(raw4)
    assert blob4 is None and stripped4 == raw4

    # absent — no-op passthrough
    raw3 = b'{"choices":[{"token_ids":[1,2]}]}'
    stripped3, blob3 = strip_routed_experts_data(raw3)
    assert blob3 is None and stripped3 == raw3
