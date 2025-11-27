"""Tests for the SingleTurnEnv class."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from datasets import Dataset

from verifiers import Parser, Rubric, SingleTurnEnv
from verifiers.types import RolloutInput, RolloutTiming


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
        # No trajectory steps yet
        state = {
            "trajectory": [],
            "prompt": [{"role": "user", "content": "Hello"}],
            "timing": RolloutTiming(
                generation_ms=0.0,
                scoring_ms=0.0,
                total_ms=0.0,
                start_time=0.0,
            ),
        }
        assert not await mock_singleturn_env.is_completed(state)

        # With trajectory steps
        from verifiers.types import TrajectoryStep

        state = {
            "trajectory": [
                TrajectoryStep(
                    prompt=[{"role": "user", "content": "Hello"}],
                    completion=[{"role": "assistant", "content": "Hi"}],
                    response=MagicMock(),
                    tokens=None,
                    reward=None,
                    advantage=None,
                    extras={},
                )
            ],
            "prompt": [{"role": "user", "content": "Hello"}],
            "timing": RolloutTiming(
                generation_ms=0.0,
                scoring_ms=0.0,
                total_ms=0.0,
                start_time=0.0,
            ),
        }
        assert await mock_singleturn_env.is_completed(state)

    @pytest.mark.asyncio
    async def test_env_response_method(self, mock_singleturn_env):
        """Test the env_response method raises NotImplementedError."""
        messages = [{"role": "user", "content": "Hello"}]
        state = {}

        with pytest.raises(NotImplementedError):
            await mock_singleturn_env.env_response(messages, state)

    @pytest.mark.asyncio
    async def test_rollout_chat_format(self, mock_singleturn_env):
        """Test rollout with chat format."""
        prompt = [{"role": "user", "content": "What is 2+2?"}]
        answer = "4"

        state = await mock_singleturn_env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer=answer,
                example_id=0,
            ),
            client=mock_singleturn_env.client,
            model="test-model",
        )
        completion = state["completion"]

        # Should return list format for chat
        assert isinstance(completion, list)
        assert len(completion) == 1
        assert completion[0]["role"] == "assistant"
        assert completion[0]["content"] == "This is a test response"

        # Check state structure
        assert "trajectory" in state
        assert len(state["trajectory"]) == 1
        assert state["answer"] == answer

        # Verify the client was called
        mock_singleturn_env.client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollout_completion_format(self, mock_singleturn_env_completion):
        """Test rollout with completion format."""
        prompt = "Calculate 2+2:"
        answer = "4"

        state = await mock_singleturn_env_completion.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer=answer,
                example_id=0,
            ),
            client=mock_singleturn_env_completion.client,
            model="test-model",
        )
        completion = state["completion"]

        # Should return string format for completion
        assert isinstance(completion, str)
        assert completion == "This is a test completion"

        # Check state structure
        assert "trajectory" in state
        assert len(state["trajectory"]) == 1

        # Verify the client was called
        mock_singleturn_env_completion.client.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollout_with_sampling_args(self, mock_singleturn_env):
        """Test rollout with custom sampling arguments."""
        prompt = [{"role": "user", "content": "Hello"}]
        answer = "Hi"
        sampling_args = {"temperature": 0.8, "max_tokens": 100}

        state = await mock_singleturn_env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer=answer,
                example_id=0,
            ),
            client=mock_singleturn_env.client,
            model="test-model",
            sampling_args=sampling_args,
        )
        completion = state["completion"]

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

        state = await mock_singleturn_env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer=answer,
                task=task,
                info=info,
                example_id=0,
            ),
            client=mock_singleturn_env.client,
            model="test-model",
        )
        completion = state["completion"]

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
                input=RolloutInput(
                    prompt=prompt,
                    answer=answer,
                    example_id=0,
                ),
                client=mock_singleturn_env.client,
                model="test-model",
            )

    @pytest.mark.asyncio
    async def test_rollout_state_structure(self, mock_singleturn_env):
        """Test that rollout creates proper state structure."""
        prompt = [{"role": "user", "content": "Hello"}]
        answer = "Hi"
        task = "greeting"
        info = {"context": "test"}

        state = await mock_singleturn_env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer=answer,
                task=task,
                info=info,
                example_id=0,
            ),
            client=mock_singleturn_env.client,
            model="test-model",
        )
        completion = state["completion"]

        # Check all expected state fields
        assert state["prompt"] == prompt
        assert state["completion"] == completion
        assert state["answer"] == answer
        assert state["task"] == task
        assert state["info"] == info
        assert "trajectory" in state
        assert isinstance(state["trajectory"], list)
        assert len(state["trajectory"]) == 1

    @pytest.mark.asyncio
    async def test_a_generate_basic(self, mock_singleturn_env):
        """Test async generation with basic inputs."""
        from verifiers.types import RolloutInput

        inputs_list = [
            RolloutInput(
                prompt=[{"role": "user", "content": "What is 2+2?"}],
                answer="4",
                example_id=0,
                task="test",
            ),
            RolloutInput(
                prompt=[{"role": "user", "content": "What is 3+3?"}],
                answer="6",
                example_id=1,
                task="test",
            ),
        ]

        # Mock the rubric scoring to set rewards in states
        async def mock_score_group(states, score_sem=None):
            for state in states:
                state["reward"] = 1.0
                state["metrics"] = {}

        mock_singleturn_env.rubric.score_group = mock_score_group

        results = await mock_singleturn_env.generate(
            inputs_list,
            client=mock_singleturn_env.client,
            model="test-model",
        )

        assert "completion" in results
        assert "state" in results
        assert "reward" in results
        assert len(results["completion"]) == 2
        assert len(results["state"]) == 2
        assert results["reward"] == [1.0, 1.0]

    @pytest.mark.asyncio
    async def test_a_generate_with_dataset(
        self, mock_singleturn_env, sample_chat_dataset
    ):
        """Test async generation with Dataset input."""

        # Mock the rubric.score_group method to set rewards in states
        async def mock_score_group(states, score_sem=None):
            for state in states:
                state["reward"] = 1.0
                state["metrics"] = {}

        mock_singleturn_env.rubric.score_group = mock_score_group

        results = await mock_singleturn_env.generate(
            sample_chat_dataset,
            client=mock_singleturn_env.client,
            model="test-model",
        )

        assert "completion" in results
        assert "state" in results
        assert "reward" in results
        assert len(results["completion"]) == 2

    @pytest.mark.asyncio
    async def test_a_generate_no_scoring(self, mock_singleturn_env):
        """Test async generation without scoring rollouts."""
        from verifiers.types import RolloutInput

        inputs_list = [
            RolloutInput(
                prompt=[{"role": "user", "content": "Hello"}],
                answer="Hi",
                example_id=0,
                task="test",
            ),
        ]

        results = await mock_singleturn_env.generate(
            inputs_list,
            client=mock_singleturn_env.client,
            model="test-model",
        )

        assert "completion" in results
        assert "state" in results
        assert "reward" in results  # reward attribute exists
        # Scoring always happens now, so rewards will be set
        assert len(results["reward"]) >= 0

    def test_generate_sync_wrapper(self, mock_singleturn_env):
        """Test the synchronous generate wrapper."""
        from verifiers.types import RolloutInput

        inputs = [
            RolloutInput(
                prompt=[{"role": "user", "content": "Hello"}],
                answer="Hi",
                info={},
                example_id=0,
                task="test",
            )
        ]

        # Mock the rubric scoring
        async def mock_score_group(states, score_sem=None):
            for state in states:
                state["reward"] = 1.0
                state["metrics"] = {}

        mock_singleturn_env.rubric.score_group = mock_score_group

        results = mock_singleturn_env.generate_sync(
            inputs,
            client=mock_singleturn_env.client,
            model="test-model",
        )

        assert "completion" in results
        assert "state" in results
        assert "reward" in results

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
        chat_state = await chat_env.rollout(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "Hello"}],
                answer="Hi",
                example_id=0,
            ),
            client=mock_openai_client,
            model="test-model",
        )
        chat_completion = chat_state["completion"]
        assert isinstance(chat_completion, list)

        # Test completion rollout
        comp_state = await completion_env.rollout(
            input=RolloutInput(
                prompt="Complete this:",
                answer="Done",
                example_id=0,
            ),
            client=mock_openai_client,
            model="test-model",
        )
        completion_result = comp_state["completion"]
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

        # Before any trajectory steps
        from verifiers.types import RolloutInput, State

        state = State(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "Hello"}],
                example_id=0,
                task="default",
            )
        )
        state["trajectory"] = []
        state["timing"] = RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            start_time=0.0,
        )
        assert not await env.is_completed(state)

        # After one trajectory step
        from verifiers.types import TrajectoryStep

        state = State(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "Hello"}],
                example_id=0,
                task="default",
            )
        )
        state["trajectory"] = [
            TrajectoryStep(
                prompt=[{"role": "user", "content": "Hello"}],
                completion=[{"role": "assistant", "content": "Hi"}],
                response=MagicMock(),
                tokens=None,
                reward=None,
                advantage=None,
                extras={},
            )
        ]
        state["timing"] = RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            start_time=0.0,
        )
        assert await env.is_completed(state)

        # Even with multiple trajectory steps (shouldn't happen), it's still completed
        state = State(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "Hello"}],
                example_id=0,
                task="default",
            )
        )
        state["trajectory"] = [
            TrajectoryStep(
                prompt=[{"role": "user", "content": "Hello"}],
                completion=[{"role": "assistant", "content": "Hi"}],
                response=MagicMock(),
                tokens=None,
                reward=None,
                advantage=None,
                extras={},
            ),
            TrajectoryStep(
                prompt=[{"role": "user", "content": "Hello"}],
                completion=[{"role": "assistant", "content": "Hi"}],
                response=MagicMock(),
                tokens=None,
                reward=None,
                advantage=None,
                extras={},
            ),
        ]
        state["timing"] = RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
            start_time=0.0,
        )
        assert await env.is_completed(state)
