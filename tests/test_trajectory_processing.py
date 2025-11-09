"""Tests for trajectory-based processing.

Covers:
- parse_response_tokens for extracting tokens from vLLM responses
- Trajectory step processing for training data
- Handling of missing token data
"""

from unittest.mock import MagicMock

import pytest

from verifiers.types import RolloutInput, State, TrajectoryStep, TrajectoryStepTokens
from verifiers.utils.response_utils import parse_response_tokens


@pytest.mark.asyncio
async def test_parse_response_tokens_chat_with_tokens():
    """Test parsing tokens from chat completion response with token data."""
    from verifiers.types import ChatCompletion

    mock_response = MagicMock(spec=ChatCompletion)
    mock_response.prompt_token_ids = [1, 2, 3]
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].token_ids = [4, 5, 6]
    mock_response.choices[0].logprobs = MagicMock()
    mock_response.choices[0].logprobs.content = [
        MagicMock(logprob=-0.1),
        MagicMock(logprob=-0.2),
        MagicMock(logprob=-0.3),
    ]

    tokens = await parse_response_tokens(mock_response, "chat")

    assert tokens is not None
    assert tokens["prompt_ids"] == [1, 2, 3]
    assert tokens["completion_ids"] == [4, 5, 6]
    assert tokens["prompt_mask"] == [0, 0, 0]
    assert tokens["completion_mask"] == [1, 1, 1]
    assert tokens["completion_logprobs"] == [-0.1, -0.2, -0.3]


@pytest.mark.asyncio
async def test_parse_response_tokens_chat_without_tokens():
    """Test parsing tokens from chat completion response without token data."""
    from verifiers.types import ChatCompletion

    mock_response = MagicMock(spec=ChatCompletion)
    mock_response.choices = [MagicMock()]
    del mock_response.prompt_token_ids

    tokens = await parse_response_tokens(mock_response, "chat")

    assert tokens is None


@pytest.mark.asyncio
async def test_parse_response_tokens_completion_with_tokens():
    """Test parsing tokens from completion response with token data."""
    from verifiers.types import Completion

    mock_response = MagicMock(spec=Completion)
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].prompt_token_ids = [10, 20]
    mock_response.choices[0].token_ids = [30, 40, 50]
    mock_response.choices[0].logprobs = MagicMock()
    mock_response.choices[0].logprobs.token_logprobs = [-0.5, -0.6, -0.7]

    tokens = await parse_response_tokens(mock_response, "completion")

    assert tokens is not None
    assert tokens["prompt_ids"] == [10, 20]
    assert tokens["completion_ids"] == [30, 40, 50]
    assert tokens["prompt_mask"] == [0, 0]
    assert tokens["completion_mask"] == [1, 1, 1]
    assert tokens["completion_logprobs"] == [-0.5, -0.6, -0.7]


@pytest.mark.asyncio
async def test_parse_response_tokens_completion_without_tokens():
    """Test parsing tokens from completion response without token data."""
    from verifiers.types import Completion

    mock_response = MagicMock(spec=Completion)
    mock_response.choices = [MagicMock()]
    del mock_response.choices[0].prompt_token_ids

    tokens = await parse_response_tokens(mock_response, "completion")

    assert tokens is None


def test_process_trajectory_steps_for_training():
    """Test processing trajectory steps into training examples."""
    state1 = State(
        input=RolloutInput(
            prompt=[{"role": "user", "content": "Hello"}],
            example_id=0,
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
            ),
            reward=1.0,
            advantage=None,
            extras={},
        )
    ]

    state2 = State(
        input=RolloutInput(
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
            ),
            reward=0.5,
            advantage=None,
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


def test_process_trajectory_steps_skip_missing_tokens():
    """Test that trajectory steps without tokens are skipped."""
    state = State(
        input=RolloutInput(
            prompt=[{"role": "user", "content": "Hello"}],
            example_id=0,
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
            ),
            reward=0.5,
            advantage=None,
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
