"""Tests for trajectory-based processing.

Covers:
- parse_response_tokens for extracting tokens from vLLM responses
- Trajectory step processing for training data
- Handling of missing token data
"""

from unittest.mock import MagicMock

import pytest

from verifiers.types import State, TrajectoryStep, TrajectoryStepTokens
from verifiers.utils.response_utils import (
    parse_response_tokens,
    parse_tool_calls_from_content,
)


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


def test_parse_tool_calls_from_content_empty():
    """parse_tool_calls_from_content returns [] for empty or non-string."""
    assert parse_tool_calls_from_content("") == []
    assert parse_tool_calls_from_content(None) == []  # type: ignore[arg-type]


def test_parse_tool_calls_from_content_hermes_style():
    """parse_tool_calls_from_content extracts Hermes-style JSON tool calls."""
    text = 'I will call the tool. {"name": "get_weather", "arguments": {"city": "Boston"}} done.'
    out = parse_tool_calls_from_content(text)
    assert len(out) == 1
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "get_weather"
    assert out[0]["function"]["arguments"] == '{"city": "Boston"}'
    assert out[0]["id"].startswith("call_")


def test_parse_tool_calls_from_content_arguments_string():
    """parse_tool_calls_from_content accepts arguments as JSON string."""
    text = '{"name": "run_code", "arguments": "{\\"code\\": \\"1+1\\"}"}'
    out = parse_tool_calls_from_content(text)
    assert len(out) == 1
    assert out[0]["function"]["name"] == "run_code"
    assert out[0]["function"]["arguments"] == '{"code": "1+1"}'


def test_parse_tool_calls_from_content_parameters_key():
    """parse_tool_calls_from_content accepts 'parameters' as well as 'arguments'."""
    text = '{"name": "search", "parameters": {"q": "test"}}'
    out = parse_tool_calls_from_content(text)
    assert len(out) == 1
    assert out[0]["function"]["name"] == "search"
    assert out[0]["function"]["arguments"] == '{"q": "test"}'


@pytest.mark.asyncio
async def test_parse_response_messages_tool_calls_from_reasoning():
    """parse_response_messages populates tool_calls from reasoning when API omits them."""
    from unittest.mock import MagicMock

    from verifiers.types import ChatCompletion

    from verifiers.utils.response_utils import parse_response_messages

    mock_message = MagicMock()
    mock_message.content = None
    mock_message.tool_calls = None
    mock_message.reasoning = 'Think step by step. {"name": "call_python_repl", "arguments": {"code": "1+1"}}'
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock(spec=ChatCompletion)
    mock_response.choices = [mock_choice]

    messages = await parse_response_messages(mock_response, "chat")
    assert isinstance(messages, list)
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "assistant"
    assert "tool_calls" in msg
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "call_python_repl"
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"code": "1+1"}'


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
