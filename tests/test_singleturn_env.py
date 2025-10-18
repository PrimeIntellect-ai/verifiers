"""Tests for the SingleTurnEnv class."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from datasets import Dataset

from verifiers import Parser, Rubric, SingleTurnEnv
from verifiers.types import RolloutScores


class TestSingleTurnEnv:
    """Test cases for the SingleTurnEnv class."""

    def test_singleturn_env_initialization_chat(
        self, mock_openai_client, sample_dataset
    ):
        """Test SingleTurnEnv initialization with chat format."""
        env = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            message_type="chat",
            system_prompt="You are helpful.",
            parser=Parser(),
            rubric=Rubric(),
        )
        assert env.message_type == "chat"
        assert isinstance(env.parser, Parser)
        assert isinstance(env.rubric, Rubric)

    def test_singleturn_env_initialization_completion(self, mock_openai_client):
        """Test SingleTurnEnv initialization with completion format."""
        completion_dataset = Dataset.from_dict(
            {
                "prompt": ["Calculate 2+2:", "What is the capital?"],
                "answer": ["4", "It depends on the country"],
            }
        )

        env = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=completion_dataset,
            message_type="completion",
            parser=Parser(),
            rubric=Rubric(),
        )
        assert env.message_type == "completion"

    @pytest.mark.asyncio
    async def test_is_completed_method(self, mock_singleturn_env):
        """Test the is_completed method logic."""
        # No responses yet
        messages = [{"role": "user", "content": "Hello"}]
        state = {"responses": []}
        assert not await mock_singleturn_env.is_completed(messages, state)

        # With responses
        state = {"responses": [MagicMock()]}
        assert await mock_singleturn_env.is_completed(messages, state)

    @pytest.mark.asyncio
    async def test_env_response_method(self, mock_singleturn_env):
        """Test the env_response method (which should never be called in practice)."""
        messages = [{"role": "user", "content": "Hello"}]
        state = {}

        response, new_state = await mock_singleturn_env.env_response(messages, state)

        # Should return minimal response (env_response returns a list of messages)
        assert len(response) == 1
        assert response[0]["role"] == "user"
        assert response[0]["content"] == ""
        assert new_state == state

    @pytest.mark.asyncio
    async def test_rollout_chat_format(self, mock_singleturn_env):
        """Test rollout with chat format."""
        prompt = [{"role": "user", "content": "What is 2+2?"}]
        answer = "4"

        completion, state = await mock_singleturn_env.rollout(
            client=mock_singleturn_env.client,
            model="test-model",
            prompt=prompt,
            answer=answer,
        )

        # Should return list format for chat
        assert isinstance(completion, list)
        assert len(completion) == 1
        assert completion[0]["role"] == "assistant"
        assert completion[0]["content"] == "This is a test response"

        # Check state structure
        assert "responses" in state
        assert len(state["responses"]) == 1
        assert state["answer"] == answer

        # Verify the client was called
        mock_singleturn_env.client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollout_completion_format(self, mock_singleturn_env_completion):
        """Test rollout with completion format."""
        prompt = "Calculate 2+2:"
        answer = "4"

        completion, state = await mock_singleturn_env_completion.rollout(
            client=mock_singleturn_env_completion.client,
            model="test-model",
            prompt=prompt,
            answer=answer,
        )

        # Should return string format for completion
        assert isinstance(completion, str)
        assert completion == "This is a test completion"

        # Check state structure
        assert "responses" in state
        assert len(state["responses"]) == 1

        # Verify the client was called
        mock_singleturn_env_completion.client.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollout_with_sampling_args(self, mock_singleturn_env):
        """Test rollout with custom sampling arguments."""
        prompt = [{"role": "user", "content": "Hello"}]
        answer = "Hi"
        sampling_args = {"temperature": 0.8, "max_tokens": 100}

        completion, state = await mock_singleturn_env.rollout(
            client=mock_singleturn_env.client,
            model="test-model",
            prompt=prompt,
            answer=answer,
            sampling_args=sampling_args,
        )

        assert isinstance(completion, list)
        assert completion[0]["content"] == "This is a test response"

        # Verify sampling args were passed
        call_args = mock_singleturn_env.client.chat.completions.create.call_args
        assert "temperature" in call_args.kwargs
        assert "max_completion_tokens" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_rollout_with_task_and_info(self, mock_singleturn_env):
        """Test rollout with task and info parameters."""
        prompt = [{"role": "user", "content": "Test question"}]
        answer = "Test answer"
        task = "math"
        info = {"difficulty": "easy"}

        completion, state = await mock_singleturn_env.rollout(
            client=mock_singleturn_env.client,
            model="test-model",
            prompt=prompt,
            answer=answer,
            task=task,
            info=info,
        )

        assert isinstance(completion, list)
        # Check state contains all the information
        assert state["answer"] == answer
        assert state["task"] == task
        assert state["info"] == info

    @pytest.mark.asyncio
    async def test_rollout_error_handling(self, mock_singleturn_env):
        """Test rollout handles errors from get_model_response."""
        # Mock get_model_response to return an error
        mock_singleturn_env.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        prompt = [{"role": "user", "content": "Hello"}]
        answer = "Hi"

        with pytest.raises(Exception, match="API Error"):
            await mock_singleturn_env.rollout(
                client=mock_singleturn_env.client,
                model="test-model",
                prompt=prompt,
                answer=answer,
            )

    @pytest.mark.asyncio
    async def test_rollout_state_structure(self, mock_singleturn_env):
        """Test that rollout creates proper state structure."""
        prompt = [{"role": "user", "content": "Hello"}]
        answer = "Hi"
        task = "greeting"
        info = {"context": "test"}

        completion, state = await mock_singleturn_env.rollout(
            client=mock_singleturn_env.client,
            model="test-model",
            prompt=prompt,
            answer=answer,
            task=task,
            info=info,
        )

        # Check all expected state fields
        assert state["prompt"] == prompt
        assert state["completion"] == completion
        assert state["answer"] == answer
        assert state["task"] == task
        assert state["info"] == info
        assert "responses" in state
        assert isinstance(state["responses"], list)
        assert len(state["responses"]) == 1

    @pytest.mark.asyncio
    async def test_a_generate_basic(self, mock_singleturn_env):
        """Test async generation with basic inputs."""
        inputs = {
            "prompt": [
                [{"role": "user", "content": "What is 2+2?"}],
                [{"role": "user", "content": "What is 3+3?"}],
            ],
            "answer": ["4", "6"],
            "example_id": [0, 1],
        }

        # Mock the rubric.score_rollouts method
        mock_singleturn_env.rubric.score_rollouts = AsyncMock(
            return_value=RolloutScores(reward=[1.0, 1.0], metrics={})
        )

        results = await mock_singleturn_env.a_generate(
            inputs,
            client=mock_singleturn_env.client,
            model="test-model",
            interleave_scoring=False,
        )

        assert hasattr(results, "completion")
        assert hasattr(results, "state")
        assert hasattr(results, "reward")
        assert len(results.completion) == 2
        assert len(results.state) == 2
        assert results.reward == [1.0, 1.0]

    @pytest.mark.asyncio
    async def test_a_generate_with_dataset(
        self, mock_singleturn_env, sample_chat_dataset
    ):
        """Test async generation with Dataset input."""
        # Mock the rubric.score_rollouts method
        mock_singleturn_env.rubric.score_rollouts = AsyncMock(
            return_value=RolloutScores(reward=[1.0, 1.0], metrics={})
        )

        results = await mock_singleturn_env.a_generate(
            sample_chat_dataset,
            client=mock_singleturn_env.client,
            model="test-model",
            interleave_scoring=False,
        )

        assert hasattr(results, "completion")
        assert hasattr(results, "state")
        assert hasattr(results, "reward")
        assert len(results.completion) == 2

    @pytest.mark.asyncio
    async def test_a_generate_no_scoring(self, mock_singleturn_env):
        """Test async generation without scoring rollouts."""
        inputs = {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "answer": ["Hi"],
            "example_id": [0],
        }

        results = await mock_singleturn_env.a_generate(
            inputs,
            client=mock_singleturn_env.client,
            model="test-model",
            score_rollouts=False,
        )

        assert hasattr(results, "completion")
        assert hasattr(results, "state")
        assert hasattr(results, "reward")  # reward attribute exists but should be empty
        assert results.reward == []  # Should be empty when score_rollouts=False

    def test_generate_sync_wrapper(self, mock_singleturn_env):
        """Test the synchronous generate wrapper."""
        inputs = {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "answer": ["Hi"],
            "info": [{}],
            "example_id": [0],
        }

        # Mock the rubric.score_rollouts method
        mock_singleturn_env.rubric.score_rollouts = AsyncMock(
            return_value=RolloutScores(reward=[1.0], metrics={})
        )

        results = mock_singleturn_env.generate_sync(
            inputs,
            client=mock_singleturn_env.client,
            model="test-model",
            interleave_scoring=False,
        )

        assert hasattr(results, "completion")
        assert hasattr(results, "state")
        assert hasattr(results, "reward")

    @pytest.mark.asyncio
    async def test_different_message_types_in_same_env(
        self, mock_openai_client, sample_dataset
    ):
        """Test that environment respects its message_type setting."""
        # Chat environment
        chat_env = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            message_type="chat",
        )

        # Completion environment
        completion_dataset = Dataset.from_dict(
            {"prompt": ["Test prompt"], "answer": ["Test answer"]}
        )
        completion_env = SingleTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=completion_dataset,
            message_type="completion",
        )

        # Test chat rollout
        chat_completion, chat_state = await chat_env.rollout(
            client=mock_openai_client,
            model="test-model",
            prompt=[{"role": "user", "content": "Hello"}],
            answer="Hi",
        )
        assert isinstance(chat_completion, list)

        # Test completion rollout
        completion_result, comp_state = await completion_env.rollout(
            client=mock_openai_client,
            model="test-model",
            prompt="Complete this:",
            answer="Done",
        )
        assert isinstance(completion_result, str)

    @pytest.mark.asyncio
    async def test_singleturn_stops_after_one_response(
        self, mock_openai_client, sample_dataset
    ):
        """Test that SingleTurnEnv truly stops after one response."""
        # We'll verify this by checking the is_completed logic
        env = SingleTurnEnv(
            client=mock_openai_client, model="test-model", dataset=sample_dataset
        )

        # Before any responses
        state = {"responses": []}
        assert not await env.is_completed([], state)

        # After one response
        state = {"responses": [MagicMock()]}
        assert await env.is_completed([], state)

        # Even with multiple responses (shouldn't happen), it's still completed
        state = {"responses": [MagicMock(), MagicMock()]}
        assert await env.is_completed([], state)
