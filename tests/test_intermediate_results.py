"""Tests for intermediate results and interleaving features."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datasets import Dataset

from verifiers import Parser, Rubric, SingleTurnEnv
from verifiers.types import RolloutScores


@pytest.mark.asyncio
async def test_intermediate_results_saving(mock_singleturn_env):
    """Test that intermediate results are saved when enabled."""
    inputs = {
        "prompt": [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is 3+3?"}],
        ],
        "answer": ["4", "6"],
    }

    # Mock the rubric.score_rollouts method
    mock_singleturn_env.rubric.score_rollouts = AsyncMock(
        return_value=RolloutScores(reward=[1.0, 1.0], metrics={})
    )

    # Enable intermediate results saving
    mock_singleturn_env.save_intermediate = True

    # Capture log messages
    with pytest.LogCapture() as log:
        results = await mock_singleturn_env.a_generate(inputs)

        # Verify intermediate results were logged
        assert "Saved intermediate result 1/2" in log.records
        assert "Saved intermediate result 2/2" in log.records

    assert len(results.completion) == 2
    assert len(results.state) == 2
    assert results.reward == [1.0, 1.0]


@pytest.mark.asyncio
async def test_interleaved_reward_computation(mock_singleturn_env):
    """Test that rewards are computed after each rollout when interleaving is enabled."""
    inputs = {
        "prompt": [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is 3+3?"}],
        ],
        "answer": ["4", "6"],
    }

    # Mock the rubric.score_rollouts method to track calls
    mock_score_rollouts = AsyncMock(
        return_value=RolloutScores(reward=[1.0], metrics={})
    )
    mock_singleturn_env.rubric.score_rollouts = mock_score_rollouts

    # Enable interleaved reward computation
    mock_singleturn_env.interleave_rewards = True
    mock_singleturn_env.save_intermediate = True

    results = await mock_singleturn_env.a_generate(inputs)

    # Verify score_rollouts was called for each example
    assert mock_score_rollouts.call_count == 2

    # Each call should have been for a single example
    for call_args in mock_score_rollouts.call_args_list:
        assert len(call_args.kwargs["prompts"]) == 1
        assert len(call_args.kwargs["completions"]) == 1
        assert len(call_args.kwargs["answers"]) == 1
        assert len(call_args.kwargs["states"]) == 1


@pytest.mark.asyncio
async def test_configuration_options(mock_openai_client, sample_dataset):
    """Test that configuration options are properly set during initialization."""
    env = SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_dataset,
        save_intermediate=True,
        interleave_rewards=True,
    )

    assert env.save_intermediate is True
    assert env.interleave_rewards is True

    # Test with default values
    env_default = SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_dataset,
    )

    assert env_default.save_intermediate is False
    assert env_default.interleave_rewards is False


@pytest.mark.asyncio
async def test_intermediate_results_disabled(mock_singleturn_env):
    """Test that intermediate results are not saved when disabled."""
    inputs = {
        "prompt": [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is 3+3?"}],
        ],
        "answer": ["4", "6"],
    }

    # Mock the rubric.score_rollouts method
    mock_singleturn_env.rubric.score_rollouts = AsyncMock(
        return_value=RolloutScores(reward=[1.0, 1.0], metrics={})
    )

    # Ensure intermediate results saving is disabled
    mock_singleturn_env.save_intermediate = False

    # Capture log messages
    with pytest.LogCapture() as log:
        results = await mock_singleturn_env.a_generate(inputs)

        # Verify no intermediate results were logged
        assert not any("Saved intermediate result" in record.msg for record in log.records)

    assert len(results.completion) == 2
    assert len(results.state) == 2
    assert results.reward == [1.0, 1.0]


@pytest.mark.asyncio
async def test_evaluate_with_intermediate_results(mock_singleturn_env, sample_dataset):
    """Test that evaluate method properly handles intermediate results."""
    # Mock the rubric.score_rollouts method
    mock_singleturn_env.rubric.score_rollouts = AsyncMock(
        return_value=RolloutScores(reward=[1.0], metrics={})
    )

    # Enable both features
    mock_singleturn_env.save_intermediate = True
    mock_singleturn_env.interleave_rewards = True

    # Set up the environment with a dataset
    mock_singleturn_env.dataset = sample_dataset

    # Capture log messages
    with pytest.LogCapture() as log:
        results = mock_singleturn_env.evaluate(
            client=mock_singleturn_env.client,
            model="test-model",
            num_examples=2,
            rollouts_per_example=1,
        )

        # Verify intermediate results were logged
        assert any("Saved intermediate result" in record.msg for record in log.records)

    assert len(results.completion) == 2
    assert len(results.state) == 2
    assert results.reward == [1.0, 1.0]
