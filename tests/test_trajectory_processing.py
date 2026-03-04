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


def test_process_trajectory_steps_for_training(make_input):
    """Test processing trajectory steps into training examples."""
    state1 = State(
        input=make_input(
            prompt=[{"role": "user", "content": "Hello"}],
            task="test",
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
            task="test",
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
            task="test",
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
