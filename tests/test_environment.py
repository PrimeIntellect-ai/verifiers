"""Tests for the base Environment class."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from datasets import Dataset

import verifiers as vf
from verifiers import Environment, Parser, Rubric, ThinkParser
from verifiers.types import (
    GenerateOutputs,
    Messages,
    Response,
    ResponseMessage,
    RolloutInput,
    SamplingArgs,
    Tool,
    ToolCall,
)
from verifiers.utils.save_utils import make_dataset as build_dataset


# Create a concrete implementation for testing the abstract base class
class SimpleEnvironment(Environment):
    """Simple implementation of Environment for testing."""

    async def setup_state(self, state):
        """Setup state for SimpleEnvironment."""
        return state

    async def rollout(
        self,
        input: RolloutInput,
        client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ):
        """Simple test rollout implementation."""
        state = await self.init_state(input, client=client, model=model)
        try:
            state = await self.setup_state(state)

            prompt_messages = state["prompt"]
            response = await self.get_model_response(state, prompt_messages)

            from verifiers.utils.response_utils import parse_response_message

            completion_messages = await parse_response_message(response)
            from verifiers.types import TrajectoryStep
            from verifiers.utils.response_utils import parse_response_tokens

            tokens = await parse_response_tokens(response)
            trajectory_step = TrajectoryStep(
                prompt=prompt_messages,
                completion=completion_messages,
                response=response,
                tokens=tokens,
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id=state["trajectory_id"],
                extras={},
            )
            state["trajectory"].append(trajectory_step)
            state["is_completed"] = True

            from verifiers.utils.message_utils import concat_messages

            last_prompt = state["trajectory"][-1]["prompt"]
            last_completion = state["trajectory"][-1]["completion"]
            full_conversation = concat_messages([last_prompt, last_completion])
            state["completion"] = full_conversation[len(state["prompt"]) :]
        except vf.Error as e:
            state["error"] = e

        return state


class TestEnvironmentBase:
    """Test cases for the base Environment class."""

    def test_environment_initialization(self, sample_dataset):
        """Test that Environment initializes correctly."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )
        assert env.message_type == "chat"
        assert isinstance(env.parser, Parser)
        assert isinstance(env.rubric, Rubric)

    def test_environment_with_eval_dataset_only(self, sample_dataset):
        """Test Environment with only eval_dataset."""
        env = SimpleEnvironment(
            eval_dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )
        assert env.dataset is None
        assert env.eval_dataset is not None

    def test_environment_with_tool_defs_initializes_tools(self, sample_dataset):
        """Test constructor-time tool_defs initialization."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
            tool_defs=[
                {
                    "name": "echo",
                    "description": "Echo text",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )
        assert env.tool_defs is not None
        assert isinstance(env.tool_defs[0], Tool)
        assert env.tool_defs[0].name == "echo"

    def test_environment_rejects_oai_tools_param(self, sample_dataset):
        """Test constructor rejects deprecated oai_tools."""
        with pytest.raises(ValueError, match="`oai_tools` is no longer supported"):
            SimpleEnvironment(
                dataset=sample_dataset,
                parser=Parser(),
                rubric=Rubric(),
                oai_tools=[  # type: ignore[call-arg]
                    {
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "description": "Echo text",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

    def test_environment_no_datasets_raises_error(self):
        """Test that Environment raises error when no datasets provided."""
        with pytest.raises(
            ValueError, match="Either dataset or eval_dataset must be provided"
        ):
            SimpleEnvironment(
                model="test-model",
                parser=Parser(),
                rubric=Rubric(),
            )

    def test_different_parser_rubric_parser_warns(self, sample_dataset):
        """Test that warning is logged when parser and rubric parser are different."""
        from unittest.mock import patch

        think_parser = ThinkParser()
        rubric = Rubric()  # Different parser class

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            _ = SimpleEnvironment(
                dataset=sample_dataset,
                parser=think_parser,
                rubric=rubric,
            )

            mock_logger.warning.assert_called_once_with(
                "The parser and rubric parser are different. This may cause unexpected behavior."
            )

    def test_get_dataset(self, sample_dataset):
        """Test dataset retrieval."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Get full dataset
        full_dataset = env.get_dataset()
        assert len(full_dataset) == 2

        # Get subset
        subset = env.get_dataset(n=1)
        assert len(subset) == 1

    @pytest.mark.asyncio
    async def test_get_model_response_chat(self, mock_client, make_input):
        """Test get_model_response with chat format."""
        env = SimpleEnvironment(
            client=mock_client,
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric(),
        )

        prompt: Messages = [{"role": "user", "content": "Hello"}]
        state = await env.init_state(
            input=make_input(prompt=prompt),
            client=mock_client,
            model="test-model",
        )
        response = await env.get_model_response(
            state,
            prompt,
        )

        # Check response structure (now returns Response model, not raw ChatCompletion)
        assert response is not None
        assert isinstance(response, Response)
        assert response.message is not None
        assert response.message.content is not None
        assert mock_client.call_count == 1

    @pytest.mark.asyncio
    async def test_get_model_response_completion(self, mock_client, make_input):
        """Test get_model_response with completion format."""
        env = SimpleEnvironment(
            client=mock_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"prompt": ["test"], "answer": ["test"]}),
            message_type="completion",
            parser=Parser(),
            rubric=Rubric(),
        )

        prompt = "Complete this:"
        state = await env.init_state(
            input=make_input(prompt=prompt),
            client=mock_client,
            model="test-model",
        )
        response = await env.get_model_response(
            state,
            prompt,
        )

        # Check response structure (now returns Response model, not raw Completion)
        assert response is not None
        assert isinstance(response, Response)
        assert response.message is not None
        assert response.message.content is not None
        assert mock_client.call_count == 1

    @pytest.mark.asyncio
    async def test_init_state_normalizes_info_tool_defs(
        self, mock_client, sample_dataset, make_input
    ):
        """Test init_state normalizes info.tool_defs into state.tool_defs."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )
        prompt: Messages = [{"role": "user", "content": "Hello"}]
        state = await env.init_state(
            input=make_input(
                prompt=prompt,
                info={
                    "tool_defs": [
                        {
                            "name": "echo",
                            "description": "Echo text",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ]
                },
            ),
            client=mock_client,
            model="test-model",
        )
        assert state["tool_defs"]
        first_tool = state["tool_defs"][0]
        assert isinstance(first_tool, Tool)
        assert first_tool.name == "echo"

    @pytest.mark.asyncio
    async def test_init_state_rejects_info_oai_tools(
        self, mock_client, sample_dataset, make_input
    ):
        """Test init_state rejects deprecated info.oai_tools."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )
        prompt: Messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(
            ValueError, match="info\\['oai_tools'\\] is no longer supported"
        ):
            await env.init_state(
                input=make_input(
                    prompt=prompt,
                    info={
                        "oai_tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "description": "Echo text",
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            }
                        ]
                    },
                ),
                client=mock_client,
                model="test-model",
            )

    @pytest.mark.asyncio
    async def test_a_generate_with_score_rollouts(
        self, mock_client, sample_dataset, make_input
    ):
        """Test async generate with scoring enabled."""
        env = SimpleEnvironment(
            client=mock_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the rubric scoring
        async def mock_score_group(states, score_sem=None):
            for state in states:
                state["reward"] = 1.0
                state["metrics"] = {}

        env.rubric.score_group = mock_score_group  # type: ignore[attr-defined]

        inputs = [make_input()]
        outputs = await env.generate(
            inputs,
            client=mock_client,
            model="test-model",
        )

        states = outputs["outputs"]
        assert len(states) == 1
        assert "completion" in states[0]
        assert "reward" in states[0]
        assert states[0]["reward"] == 1.0

    def test_generate_sync_wrapper(self, mock_client, sample_dataset, make_input):
        """Test synchronous generate wrapper."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the rubric scoring
        async def mock_score_group(states, score_sem=None):
            for state in states:
                state["reward"] = 1.0
                state["metrics"] = {}

        env.rubric.score_group = mock_score_group  # type: ignore[attr-defined]

        inputs = [make_input()]
        outputs = env.generate_sync(
            inputs,
            client=mock_client,
            model="test-model",
        )

        states = outputs["outputs"]
        assert len(states) == 1
        assert "completion" in states[0]
        assert "reward" in states[0]
        assert states[0]["reward"] == 1.0

    def test_make_dataset(self, make_metadata, make_output):
        """Test creating a dataset from evaluation results."""

        results = GenerateOutputs(outputs=[make_output()], metadata=make_metadata())
        dataset = build_dataset(results)

        assert len(dataset) == 1
        assert "prompt" in dataset.column_names
        assert "completion" in dataset.column_names
        assert "reward" in dataset.column_names
        assert "task" in dataset.column_names
        assert "example_id" in dataset.column_names
        assert "foo" in dataset.column_names  # custom field from make_output fixture

    @pytest.mark.asyncio
    async def test_generate_updates_metadata(self, mock_client):
        """Test that metadata fields are updated after generate() completes."""
        dataset = Dataset.from_dict(
            {
                "question": ["What is 2+2?", "What is 3+3?"],
                "answer": ["4", "6"],
            }
        )

        def reward_a(**kwargs):
            return 1.0

        def reward_b(**kwargs):
            return 0.5

        env = SimpleEnvironment(
            dataset=dataset,
            rubric=Rubric(
                funcs=[reward_a, reward_b],
                weights=[0.5, 0.5],
            ),
        )

        results = await env.generate(
            inputs=env.get_dataset(n=2),
            client=mock_client,
            model="test-model",
        )

        assert results["metadata"]["time_ms"] > 0.0
        assert results["metadata"]["avg_reward"] == 0.75
        assert len(results["metadata"]["avg_metrics"]) == 2
        assert "reward_a" in results["metadata"]["avg_metrics"]
        assert "reward_b" in results["metadata"]["avg_metrics"]
        assert results["metadata"]["avg_metrics"]["reward_a"] == 1.0
        assert results["metadata"]["avg_metrics"]["reward_b"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_metadata_without_scoring(self, mock_client):
        """Test that metadata handles scoring correctly."""
        dataset = Dataset.from_dict(
            {
                "question": ["What is 2+2?"],
                "answer": ["4"],
            }
        )

        env = SimpleEnvironment(dataset=dataset, rubric=Rubric())

        # Mock scoring to return no rewards for this test
        async def mock_score_group(states, score_sem=None):
            for state in states:
                state["reward"] = 0.0
                state["metrics"] = {}

        env.rubric.score_group = mock_score_group  # type: ignore[attr-defined]

        results = await env.generate(
            inputs=env.get_dataset(n=1),
            client=mock_client,
            model="test-model",
        )

        assert results["metadata"]["time_ms"] > 0.0
        # Scoring always happens now, so rewards will be set by score_group
        # If score_group doesn't set rewards, they'll be None/0
        assert results["metadata"]["avg_reward"] >= 0.0


class TestRenderStopErrorHandling:
    """Test cases for _render_stop error handling paths."""

    @pytest.mark.asyncio
    async def test_render_stop_with_vf_error(
        self, mock_client, sample_dataset, make_input
    ):
        """Test that _render_stop logs correctly for vf.Error with cause."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        cause = ValueError("underlying cause")
        error = vf.ToolCallError()
        error.__cause__ = cause

        state = await env.init_state(
            input=make_input(prompt=[{"role": "user", "content": "test"}]),
            client=mock_client,
            model="test-model",
        )
        state["error"] = error

        async def has_error(state):
            return state.get("error") is not None

        has_error.__name__ = "has_error"

        with patch.object(env.logger, "error") as mock_logger_error:
            result = await env._render_stop(state, has_error)

            assert result is True
            assert state["stop_condition"] == "has_error"
            mock_logger_error.assert_called_once()
            call_args = mock_logger_error.call_args[0][0]
            assert "ToolCallError" in call_args

    @pytest.mark.asyncio
    async def test_render_stop_with_regular_exception(
        self, mock_client, sample_dataset, make_input
    ):
        """Test that _render_stop logs correctly for regular exceptions without cause."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        error = RuntimeError("something went wrong")

        state = await env.init_state(
            input=make_input(),
            client=mock_client,
            model="test-model",
        )
        state["error"] = error

        async def has_error(state):
            return state.get("error") is not None

        has_error.__name__ = "has_error"

        with patch.object(env.logger, "error") as mock_logger_error:
            result = await env._render_stop(state, has_error)

            assert result is True
            assert state["stop_condition"] == "has_error"
            mock_logger_error.assert_called_once()
            call_args = mock_logger_error.call_args[0][0]
            assert "RuntimeError" in call_args
            assert "caused by" not in call_args
            assert "something went wrong" in call_args


class RetryCounterEnv(SimpleEnvironment):
    """Environment that fails first N times with configurable error type."""

    def __init__(self, fail_count: int, error_type: type = vf.InfraError, **kwargs):
        super().__init__(**kwargs)
        self.fail_count = fail_count
        self.error_type = error_type
        self.call_counts: dict[int, int] = {}

    async def setup_state(self, state, **kwargs):
        example_id = state["example_id"]
        self.call_counts.setdefault(example_id, 0)
        self.call_counts[example_id] += 1

        if self.call_counts[example_id] <= self.fail_count:
            raise self.error_type(
                f"Simulated failure {self.call_counts[example_id]}/{self.fail_count}"
            )

        return state


class TestMaybeRetry:
    """Test cases for maybe_retry functionality in Environment.generate()."""

    @pytest.mark.asyncio
    async def test_retry_after_retryable_error(self, mock_client, make_input):
        """Retry after error on first 2 attempts, succeeds on 3rd with max_retries=3."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RetryCounterEnv(
            fail_count=2, dataset=dataset, parser=Parser(), rubric=Rubric()
        )

        inputs = [make_input()]
        outputs = await env.generate(
            inputs, client=mock_client, model="test-model", max_retries=3
        )
        states = outputs["outputs"]

        assert states[0].get("error") is None
        assert env.call_counts[0] == 3

    @pytest.mark.asyncio
    async def test_no_retry_after_non_retryable_error(self, mock_client, make_input):
        """Non-retryable error type is NOT retried even with max_retries > 0."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RetryCounterEnv(
            fail_count=10,
            error_type=vf.ToolError,
            dataset=dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        inputs = [make_input()]
        outputs = await env.generate(
            inputs, client=mock_client, model="test-model", max_retries=3
        )

        rollout_outputs = outputs["outputs"]
        assert env.call_counts[0] == 1  # No retries for non-retryable error
        assert rollout_outputs[0].get("error") is not None
        error_info = rollout_outputs[0]["error"]
        assert "ToolError" == error_info["error"]

    @pytest.mark.asyncio
    async def test_error_in_state_after_max_retries_exhausted(
        self, mock_client, make_input
    ):
        """Error persists in state after all retries exhausted."""
        dataset = Dataset.from_dict({"question": ["test"], "answer": ["test"]})
        env = RetryCounterEnv(
            fail_count=10, dataset=dataset, parser=Parser(), rubric=Rubric()
        )

        inputs = [make_input()]
        outputs = await env.generate(
            inputs, client=mock_client, model="test-model", max_retries=2
        )

        rollout_outputs = outputs["outputs"]
        assert env.call_counts[0] == 3  # 1 initial + 2 retries
        assert rollout_outputs[0].get("error") is not None
        error_info = rollout_outputs[0]["error"]
        assert "InfraError" == error_info["error"]


class TestEmptyModelResponseErrors:
    """Test cases for empty and invalid model response error handling."""

    @pytest.mark.asyncio
    async def test_none_response_raises_empty_model_response_error(
        self, mock_client, sample_dataset, make_input
    ):
        """Test that None response raises EmptyModelResponseError."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the client to raise EmptyModelResponseError
        mock_client.get_response = AsyncMock(
            side_effect=vf.EmptyModelResponseError("None response")
        )

        prompt: Messages = [{"role": "user", "content": "Hello"}]
        state = await env.init_state(
            input=make_input(prompt=prompt),
            client=mock_client,
            model="test-model",
        )

        with pytest.raises(vf.EmptyModelResponseError):
            await env.get_model_response(state, prompt)

    @pytest.mark.asyncio
    async def test_none_choices_raises_empty_model_response_error(
        self, mock_client, sample_dataset, make_input
    ):
        """Test that response with None choices raises EmptyModelResponseError."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the client to raise EmptyModelResponseError
        mock_client.get_response = AsyncMock(
            side_effect=vf.EmptyModelResponseError("Response has no choices")
        )

        prompt: Messages = [{"role": "user", "content": "Hello"}]
        state = await env.init_state(
            input=make_input(prompt=prompt),
            client=mock_client,
            model="test-model",
        )

        with pytest.raises(vf.EmptyModelResponseError):
            await env.get_model_response(state, prompt)

    @pytest.mark.asyncio
    async def test_wrong_number_of_choices_raises_invalid_model_response_error(
        self, mock_client, sample_dataset, make_input
    ):
        """Test that response with != 1 choices raises InvalidModelResponseError."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the client to raise InvalidModelResponseError
        mock_client.get_response = AsyncMock(
            side_effect=vf.InvalidModelResponseError("Expected 1 choice, got 2")
        )

        prompt: Messages = [{"role": "user", "content": "Hello"}]
        state = await env.init_state(
            input=make_input(prompt=prompt),
            client=mock_client,
            model="test-model",
        )

        with pytest.raises(vf.InvalidModelResponseError):
            await env.get_model_response(state, prompt)

    @pytest.mark.asyncio
    async def test_empty_choices_raises_invalid_model_response_error(
        self, mock_client, sample_dataset, make_input
    ):
        """Test that response with empty choices list raises InvalidModelResponseError."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the client to raise InvalidModelResponseError
        mock_client.get_response = AsyncMock(
            side_effect=vf.InvalidModelResponseError("Expected 1 choice, got 0")
        )

        prompt: Messages = [{"role": "user", "content": "Hello"}]
        state = await env.init_state(
            input=make_input(prompt=prompt),
            client=mock_client,
            model="test-model",
        )

        with pytest.raises(vf.InvalidModelResponseError):
            await env.get_model_response(state, prompt)

    @pytest.mark.asyncio
    async def test_chat_empty_content_no_tool_calls_raises_empty_model_response_error(
        self, mock_client, sample_dataset, make_input
    ):
        """Test that chat response with no content and no tool_calls raises EmptyModelResponseError."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the client to raise EmptyModelResponseError
        mock_client.get_response = AsyncMock(
            side_effect=vf.EmptyModelResponseError("No content and no tool calls")
        )

        prompt: Messages = [{"role": "user", "content": "Hello"}]
        state = await env.init_state(
            input=make_input(prompt=prompt),
            client=mock_client,
            model="test-model",
        )

        with pytest.raises(vf.EmptyModelResponseError):
            await env.get_model_response(state, prompt)

    @pytest.mark.asyncio
    async def test_chat_empty_string_content_no_tool_calls_raises_empty_model_response_error(
        self, mock_client, sample_dataset, make_input
    ):
        """Test that chat response with empty string content and no tool_calls raises EmptyModelResponseError."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the client to raise EmptyModelResponseError
        mock_client.get_response = AsyncMock(
            side_effect=vf.EmptyModelResponseError("Empty content and no tool calls")
        )

        prompt: Messages = [{"role": "user", "content": "Hello"}]
        state = await env.init_state(
            input=make_input(prompt=prompt),
            client=mock_client,
            model="test-model",
        )

        with pytest.raises(vf.EmptyModelResponseError):
            await env.get_model_response(state, prompt)

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls_but_no_content_succeeds(
        self, mock_client, sample_dataset, make_input
    ):
        """Test that chat response with tool_calls but no content does NOT raise error."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the client to return a Response with tool calls but no content
        from verifiers.types import Response

        mock_client.get_response = AsyncMock(
            return_value=Response(
                id="test-id",
                created=0,
                model="test-model",
                usage=None,
                message=ResponseMessage(
                    content=None,
                    reasoning_content=None,
                    finish_reason="stop",
                    is_truncated=False,
                    tokens=None,
                    tool_calls=[
                        ToolCall(id="call_123", name="test_function", arguments="{}")
                    ],
                ),
            )
        )

        prompt: Messages = [{"role": "user", "content": "Hello"}]
        state = await env.init_state(
            input=make_input(prompt=prompt),
            client=mock_client,
            model="test-model",
        )

        # Should not raise
        response = await env.get_model_response(state, prompt)
        assert response is not None
        assert response.message.tool_calls is not None

    @pytest.mark.asyncio
    async def test_completion_empty_text_raises_empty_model_response_error(
        self, mock_client, make_input
    ):
        """Test that completion response with empty text raises EmptyModelResponseError."""

        env = SimpleEnvironment(
            eval_dataset=Dataset.from_dict({"prompt": ["test"], "answer": ["test"]}),
            message_type="completion",
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the client to raise EmptyModelResponseError
        mock_client.get_response = AsyncMock(
            side_effect=vf.EmptyModelResponseError("Empty completion text")
        )

        prompt = "Complete this:"
        state = await env.init_state(
            input=make_input(prompt=prompt),
            client=mock_client,
            model="test-model",
        )

        with pytest.raises(vf.EmptyModelResponseError):
            await env.get_model_response(state, prompt)

    @pytest.mark.asyncio
    async def test_completion_none_text_raises_empty_model_response_error(
        self, mock_client, make_input
    ):
        """Test that completion response with None text raises EmptyModelResponseError."""

        env = SimpleEnvironment(
            eval_dataset=Dataset.from_dict({"prompt": ["test"], "answer": ["test"]}),
            message_type="completion",
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the client to raise EmptyModelResponseError
        mock_client.get_response = AsyncMock(
            side_effect=vf.EmptyModelResponseError("None completion text")
        )

        prompt = "Complete this:"
        state = await env.init_state(
            input=make_input(prompt=prompt),
            client=mock_client,
            model="test-model",
        )

        with pytest.raises(vf.EmptyModelResponseError):
            await env.get_model_response(state, prompt)
