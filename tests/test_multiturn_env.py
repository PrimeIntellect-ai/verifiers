"""Tests for the MultiTurnEnv class."""

import pytest
from datasets import Dataset

from verifiers import Messages, MultiTurnEnv, Parser, Rubric, State, stop
from verifiers.types import RolloutInput


class TestMultiTurnEnv:
    """Test cases for the MultiTurnEnv class."""

    def test_multiturn_env_initialization(self, mock_multiturn_env):
        """Test MultiTurnEnv initialization."""
        assert mock_multiturn_env.max_turns == 3
        assert mock_multiturn_env.message_type == "chat"  # Default from parent

    def test_multiturn_env_default_max_turns(
        self, mock_openai_client, sample_chat_dataset
    ):
        """Test MultiTurnEnv default max_turns value."""
        from tests.conftest import SimpleMultiTurnEnv

        env = SimpleMultiTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )
        assert env.max_turns == -1  # Default value

    @pytest.mark.asyncio
    async def test_basic_multiturn_rollout(self, mock_multiturn_env):
        """Test basic multi-turn conversation that completes normally."""
        # Configure mock to return responses that lead to completion
        prompt = [{"role": "user", "content": "Start conversation"}]

        # Set up responses for the conversation turns
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Start conversation"}],
            response="First response",
        )
        mock_multiturn_env.client.add_chat_response(
            messages=[
                {"role": "user", "content": "Start conversation"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Continue (turn 1)"},
            ],
            response="Second response",
        )
        mock_multiturn_env.client.add_chat_response(
            messages=[
                {"role": "user", "content": "Start conversation"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Continue (turn 1)"},
                {"role": "assistant", "content": "Second response"},
                {"role": "user", "content": "Please finish with DONE"},
            ],
            response="Final response DONE",
        )

        state = await mock_multiturn_env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer="target_answer",
                example_id=0,
            ),
            client=mock_multiturn_env.client,
            model="test-model",
        )

        # Should have: assistant + user + assistant + user + assistant
        assert len(state["trajectory"]) == 3  # Three trajectory steps
        assert state["is_completed"] is True
        assert "completion" in state  # Completion is set when is_completed returns True
        completion = state["completion"]
        assert len(completion) == 5
        assert completion[0]["role"] == "assistant"
        assert completion[0]["content"] == "First response"
        assert completion[1]["role"] == "user"
        assert completion[2]["role"] == "assistant"
        assert completion[2]["content"] == "Second response"
        assert completion[4]["content"] == "Final response DONE"

        # Check state structure
        assert state["answer"] == "target_answer"
        assert state["prompt"] == prompt

    @pytest.mark.asyncio
    async def test_max_turns_limiting(self, mock_multiturn_env_max_turns):
        """Test that rollout stops at max_turns."""
        # Set up responses that would continue indefinitely
        mock_multiturn_env_max_turns.client.set_default_responses(
            chat_response="Keep going"
        )

        prompt = [{"role": "user", "content": "Start conversation"}]
        state = await mock_multiturn_env_max_turns.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer="target_answer",
                example_id=0,
            ),
            client=mock_multiturn_env_max_turns.client,
            model="test-model",
        )

        # Should stop at max_turns=2: assistant + user + assistant (3 messages)
        assert len(state["trajectory"]) == 2  # Two trajectory steps
        assert state["is_completed"] is True
        assert "completion" in state  # Completion is set when is_completed returns True
        completion = state["completion"]
        assert len(completion) == 3
        assert completion[0]["role"] == "assistant"
        assert completion[1]["role"] == "user"
        assert completion[2]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_override_is_completed_respects_max_turns(
        self, mock_openai_client, sample_chat_dataset
    ):
        """Ensure custom is_completed implementations still honor max_turns."""

        class MaxTurnsAwareEnv(MultiTurnEnv):
            def __init__(self, **kwargs):
                super().__init__(max_turns=2, **kwargs)

            @stop
            async def done_condition(self, state: State) -> bool:
                return state.get("done", False)

            async def env_response(self, messages, state, **kwargs):  # type: ignore[override]
                state["env_calls"] = state.get("env_calls", 0) + 1
                return [
                    {
                        "role": "user",
                        "content": "Environment follow-up",
                    }
                ]

        env = MaxTurnsAwareEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        mock_openai_client.set_default_responses(chat_response="Still thinking")

        prompt: Messages = [{"role": "user", "content": "Start"}]
        state = await env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer="target",
                example_id=0,
            ),
            client=mock_openai_client,
            model="test-model",
        )
        completion = state["completion"]

        assert state.get("env_calls", 0) == 1
        assert len(state["trajectory"]) == 2
        assert isinstance(completion, list)
        assert "content" in completion[-1]
        assert completion[-1]["role"] == "assistant"
        assert completion[-1]["content"] == "Still thinking"

    @pytest.mark.asyncio
    async def test_state_initialization(self, mock_multiturn_env):
        """Test that state is properly initialized with all required fields."""
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Test state"}], response="Quick DONE"
        )

        prompt = [{"role": "user", "content": "Test state"}]
        state = await mock_multiturn_env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer="test_answer",
                task="test_task",
                info={"extra": "data"},
                example_id=0,
            ),
            client=mock_multiturn_env.client,
            model="test-model",
        )

        # Check all state fields are initialized
        assert state["prompt"] == prompt
        assert state["answer"] == "test_answer"
        assert state["task"] == "test_task"
        assert state["info"] == {"extra": "data"}
        assert state["example_id"] == 0
        assert "trajectory" in state
        assert isinstance(state["trajectory"], list)
        assert len(state["trajectory"]) > 0  # Should have at least one step
        assert state["is_completed"] is True
        assert "completion" in state  # Completion is set when is_completed returns True

    @pytest.mark.asyncio
    async def test_immediate_completion(self, mock_multiturn_env):
        """Test completion detection on first turn."""
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Quick question"}],
            response="Immediate DONE",
        )

        prompt = [{"role": "user", "content": "Quick question"}]
        state = await mock_multiturn_env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer="target_answer",
                example_id=0,
            ),
            client=mock_multiturn_env.client,
            model="test-model",
        )

        # Should complete immediately
        assert len(state["trajectory"]) == 1
        assert state["is_completed"] is True
        assert "completion" in state  # Completion is set when is_completed returns True
        completion = state["completion"]
        assert len(completion) == 1
        assert completion[0]["content"] == "Immediate DONE"

    @pytest.mark.asyncio
    async def test_env_response_integration(self, mock_multiturn_env):
        """Test that environment responses are properly integrated."""
        # Set up responses for the conversation turns
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Start conversation"}],
            response="First response",
        )
        mock_multiturn_env.client.add_chat_response(
            messages=[
                {"role": "user", "content": "Start conversation"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Continue (turn 1)"},
            ],
            response="Final response DONE",
        )

        prompt = [{"role": "user", "content": "Start conversation"}]
        state = await mock_multiturn_env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer="target_answer",
                example_id=0,
            ),
            client=mock_multiturn_env.client,
            model="test-model",
        )

        # Verify environment responses are included
        assert len(state["trajectory"]) > 0
        assert state["is_completed"] is True
        assert "completion" in state  # Completion is set when is_completed returns True
        completion = state["completion"]
        assert len(completion) >= 3
        user_messages = [msg for msg in completion if msg["role"] == "user"]
        assert len(user_messages) >= 1
        assert "Continue (turn 1)" in user_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_prompt_copying(self, mock_multiturn_env):
        """Test that original prompt is not modified."""
        original_prompt = [{"role": "user", "content": "Original message"}]
        prompt_copy = [{"role": "user", "content": "Original message"}]

        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Original message"}],
            response="Response DONE",
        )

        await mock_multiturn_env.rollout(
            input=RolloutInput(
                prompt=original_prompt,
                answer="test_answer",
                example_id=0,
            ),
            client=mock_multiturn_env.client,
            model="test-model",
        )

        # Original prompt should be unchanged
        assert original_prompt == prompt_copy

    @pytest.mark.asyncio
    async def test_sampling_args_passed_through(self, mock_multiturn_env):
        """Test that sampling arguments are passed to model calls."""
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Test sampling"}],
            response="Quick DONE",
        )

        prompt = [{"role": "user", "content": "Test sampling"}]
        sampling_args = {"temperature": 0.8, "max_tokens": 50}

        await mock_multiturn_env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer="test_answer",
                example_id=0,
            ),
            client=mock_multiturn_env.client,
            model="test-model",
            sampling_args=sampling_args,
        )

        # Verify sampling args were passed
        call_args = mock_multiturn_env.client.chat.completions.create.call_args
        assert "temperature" in call_args.kwargs
        assert "max_completion_tokens" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_completion_format_multiturn(self, mock_openai_client):
        """Test MultiTurnEnv with completion format."""

        class CompletionMultiTurnEnv(MultiTurnEnv):
            def __init__(self, **kwargs):
                super().__init__(message_type="completion", **kwargs)

            @stop
            async def done_condition(self, state: State) -> bool:
                if state["trajectory"]:
                    last_completion = state["trajectory"][-1]["completion"]
                    if isinstance(last_completion, str):
                        return "DONE" in last_completion
                return False

            async def env_response(
                self, messages: Messages, state: State, **kwargs
            ) -> Messages:
                return " Continue."

        completion_dataset = Dataset.from_dict(
            {"prompt": ["Start:"], "answer": ["Done"]}
        )

        env = CompletionMultiTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=completion_dataset,
            max_turns=3,
        )

        mock_openai_client.add_text_response("Start:", "First response")
        mock_openai_client.add_text_response(
            "Start:First response Continue.", "Final DONE"
        )

        prompt = "Start:"
        state = await env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer="Done",
                example_id=0,
            ),
            client=mock_openai_client,
            model="test-model",
        )

        assert len(state["trajectory"]) == 2
        # With max_turns=3, the rollout should complete after 2 turns
        # (max_turns_reached stop condition should trigger)
        assert state["is_completed"] is True
        assert "completion" in state  # Completion is set when is_completed returns True
        completion = state["completion"]
        assert isinstance(completion, str)
        assert "First response" in completion
        assert "DONE" in completion

    @pytest.mark.asyncio
    async def test_environment_response_state_modification(
        self, mock_openai_client, sample_chat_dataset
    ):
        """Test that environment can modify state between turns."""

        class StatefulMultiTurnEnv(MultiTurnEnv):
            @stop
            async def turn_count_reached(self, state: State) -> bool:
                return state.get("turn_count", 0) >= 2

            async def env_response(self, messages, state, **kwargs):  # type: ignore
                state["turn_count"] = state.get("turn_count", 0) + 1
                return [{"role": "user", "content": f"Turn {state['turn_count']}"}]

        env = StatefulMultiTurnEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
            max_turns=5,
            parser=Parser(),
            rubric=Rubric(),
        )

        env.client.set_default_responses(chat_response="Continue")  # type: ignore

        prompt = [{"role": "user", "content": "Start"}]
        state = await env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer="test",
                example_id=0,
            ),
            client=env.client,  # type: ignore
            model="test-model",
        )

        # Should complete when turn_count reaches 2
        assert state["turn_count"] == 2
        assert len(state["trajectory"]) >= 2
        assert state["is_completed"] is True
        assert "completion" in state  # Completion is set when is_completed returns True
        completion = state["completion"]
        assert len(completion) >= 3  # Multiple turns with env responses

    def test_abstract_methods_not_implemented(self):
        """Test that MultiTurnEnv cannot be instantiated directly (abstract class)."""
        # MultiTurnEnv is abstract and should not be instantiable without implementing abstract methods
        with pytest.raises(TypeError):
            # This should fail because MultiTurnEnv has abstract methods
            MultiTurnEnv(model="test-model", parser=Parser(), rubric=Rubric())  # type: ignore

    @pytest.mark.asyncio
    async def test_completion_detection_before_env_response(
        self, mock_openai_client, sample_chat_dataset
    ):
        """Test completion detection works before env_response is called."""

        class ImmediateCompletionEnv(MultiTurnEnv):
            @stop
            async def has_trajectory_step(self, state: State) -> bool:
                # Complete if we have any trajectory step
                return len(state["trajectory"]) > 0

            async def env_response(
                self, messages: Messages, state: State, **kwargs
            ) -> Messages:  # type: ignore
                # This should never be called due to immediate completion
                return [{"role": "user", "content": "Should not appear"}]

        env = ImmediateCompletionEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
            max_turns=5,
            parser=Parser(),
            rubric=Rubric(),
        )

        env.client.add_chat_response(  # type: ignore
            messages=[{"role": "user", "content": "Start"}], response="First response"
        )

        prompt = [{"role": "user", "content": "Start"}]
        state = await env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer="test",
                example_id=0,
            ),
            client=env.client,  # type: ignore
            model="test-model",
        )

        # Should complete immediately after first assistant response
        assert len(state["trajectory"]) == 1
        assert state["is_completed"] is True
        assert "completion" in state  # Completion is set when is_completed returns True
        completion = state["completion"]
        assert len(completion) == 1
        assert completion[0]["role"] == "assistant"  # type: ignore
        assert completion[0]["content"] == "First response"  # type: ignore

    @pytest.mark.asyncio
    async def test_responses_stored_in_state(self, mock_multiturn_env):
        """Test that model responses are stored in state['responses']."""
        # Set up a multi-turn conversation
        mock_multiturn_env.client.add_chat_response(
            messages=[{"role": "user", "content": "Start"}], response="First"
        )
        mock_multiturn_env.client.add_chat_response(
            messages=[
                {"role": "user", "content": "Start"},
                {"role": "assistant", "content": "First"},
                {"role": "user", "content": "Continue (turn 1)"},
            ],
            response="Second",
        )
        mock_multiturn_env.client.add_chat_response(
            messages=[
                {"role": "user", "content": "Start"},
                {"role": "assistant", "content": "First"},
                {"role": "user", "content": "Continue (turn 1)"},
                {"role": "assistant", "content": "Second"},
                {"role": "user", "content": "Please finish with DONE"},
            ],
            response="DONE",
        )

        prompt = [{"role": "user", "content": "Start"}]
        state = await mock_multiturn_env.rollout(
            input=RolloutInput(
                prompt=prompt,
                answer="test",
                example_id=0,
            ),
            client=mock_multiturn_env.client,
            model="test-model",
        )

        # Check that all trajectory steps are stored
        assert len(state["trajectory"]) == 3
        # Each trajectory step should have the structure returned by get_model_response
        for step in state["trajectory"]:
            assert hasattr(step["response"], "choices")
            assert len(step["response"].choices) > 0
