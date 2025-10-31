"""Tests for the base Environment class."""

from pathlib import Path
from typing import List, cast
from unittest.mock import AsyncMock, Mock

import pytest
from datasets import Dataset
from openai.types.chat.chat_completion import Choice

from verifiers import Environment, Parser, Rubric, ThinkParser
from verifiers.types import (
    ChatCompletion,
    ChatMessage,
    Completion,
    GenerateInputs,
    GenerateMetadata,
    GenerateOutputs,
    Messages,
    RolloutScores,
)
from verifiers.utils.eval_utils import make_dataset as build_dataset
from verifiers.utils.processing_utils import (
    parse_chat_completion_logprobs,
    parse_chat_completion_tokens,
    process_chat_format_vllm,
    process_completion_format_vllm,
)


# Create a concrete implementation for testing the abstract base class
class SimpleEnvironment(Environment):
    """Simple implementation of Environment for testing."""

    async def rollout(
        self,
        client,
        model,
        prompt,
        completion=None,
        answer: str = "",
        state: dict | None = None,
        task: str = "default",
        info: dict | None = None,
        example_id: int = 0,
        sampling_args: dict | None = None,
        **kwargs,
    ):
        """Simple test rollout implementation."""
        info = info or {}
        if completion is None:
            completion = await self.init_completion()
        if state is None:
            state = await self.init_state(
                prompt=prompt,
                completion=completion,
                answer=answer,
                task=task,
                info=info,
                example_id=example_id,
            )
        response = await self.get_model_response(
            prompt=prompt, client=client, model=model, sampling_args=sampling_args or {}
        )
        if self.message_type == "chat":
            assert isinstance(response, ChatCompletion)
            assert isinstance(completion, list)
            state.setdefault("responses", [])
            state["responses"].append(response)
            message = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
            completion.append(cast(ChatMessage, message))
            state["completion"] = completion
        else:
            assert isinstance(response, Completion)
            assert isinstance(completion, str)
            state.setdefault("responses", [])
            state["responses"].append(response)
            completion = completion + (response.choices[0].text or "")
            state["completion"] = completion
        return completion, state


def _make_metadata(
    num_examples: int, rollouts_per_example: int = 1
) -> GenerateMetadata:
    return GenerateMetadata(
        env_id="test-env",
        env_args={},
        model="test-model",
        base_url="http://localhost",
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args={},
        date="1970-01-01",
        time_ms=0.0,
        avg_reward=0.0,
        avg_metrics={},
        state_columns=["custom_field"],
        path_to_save=Path("test.jsonl"),
    )


class TestEnvironmentBase:
    """Test cases for the base Environment class."""

    def test_environment_initialization(self, mock_openai_client, sample_dataset):
        """Test that Environment initializes correctly."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )
        assert env.message_type == "chat"
        assert isinstance(env.parser, Parser)
        assert isinstance(env.rubric, Rubric)

    def test_environment_with_eval_dataset_only(
        self, mock_openai_client, sample_dataset
    ):
        """Test Environment with only eval_dataset."""
        env = SimpleEnvironment(
            eval_dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )
        assert env.dataset is None
        assert env.eval_dataset is not None

    def test_environment_no_datasets_raises_error(self, mock_openai_client):
        """Test that Environment raises error when no datasets provided."""
        with pytest.raises(
            ValueError, match="Either dataset or eval_dataset must be provided"
        ):
            SimpleEnvironment(
                client=mock_openai_client,
                model="test-model",
                parser=Parser(),
                rubric=Rubric(),
            )

    def test_completion_mode_with_system_prompt_raises_error(
        self, mock_openai_client, sample_dataset
    ):
        """Test that completion mode with system prompt raises error."""
        with pytest.raises(ValueError, match="not supported for completion tasks"):
            SimpleEnvironment(
                dataset=sample_dataset,
                message_type="completion",
                system_prompt="test prompt",
                parser=Parser(),
                rubric=Rubric(),
            )

    def test_different_parser_rubric_parser_warns(
        self, mock_openai_client, sample_dataset
    ):
        """Test that warning is logged when parser and rubric parser are different."""
        from unittest.mock import Mock, patch

        think_parser = ThinkParser()
        rubric = Rubric()  # Different parser class

        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            _ = SimpleEnvironment(
                client=mock_openai_client,
                model="test-model",
                dataset=sample_dataset,
                parser=think_parser,
                rubric=rubric,
            )

            mock_logger.warning.assert_called_once_with(
                "The parser and rubric parser are different. This may cause unexpected behavior."
            )

    def test_format_prompt(self, mock_openai_client, sample_dataset):
        """Test prompt formatting."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        prompt = "What is 2+2?"
        system_prompt = "You are a helpful assistant."
        few_shot: List[ChatMessage] = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]

        formatted = env.format_prompt(prompt, system_prompt, few_shot)

        assert len(formatted) == 4
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == system_prompt
        assert formatted[1]["role"] == "user"
        assert formatted[1]["content"] == "What is 1+1?"
        assert formatted[2]["role"] == "assistant"
        assert "content" in formatted[2]
        assert formatted[2]["content"] == "2"
        assert formatted[3]["role"] == "user"
        assert formatted[3]["content"] == prompt

    def test_get_dataset(self, mock_openai_client, sample_dataset):
        """Test dataset retrieval."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
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
    async def test_get_model_response_chat(self, mock_openai_client):
        """Test get_model_response with chat format."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric(),
        )

        prompt: Messages = [{"role": "user", "content": "Hello"}]
        response = await env.get_model_response(
            prompt=prompt,
            client=mock_openai_client,
            model="test-model",
            message_type="chat",
        )

        # Check response structure
        assert hasattr(response, "choices")
        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0] is not None
        assert isinstance(response.choices[0], Choice)
        assert hasattr(response.choices[0], "message")
        assert response.choices[0].message is not None
        assert hasattr(response.choices[0].message, "content")
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_model_response_completion(self, mock_openai_client):
        """Test get_model_response with completion format."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"prompt": ["test"], "answer": ["test"]}),
            message_type="completion",
            parser=Parser(),
            rubric=Rubric(),
        )

        prompt = "Complete this:"
        response = await env.get_model_response(
            prompt=prompt,
            client=mock_openai_client,
            model="test-model",
            message_type="completion",
        )

        # Check response structure
        assert hasattr(response, "choices")
        assert response is not None
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "text")
        mock_openai_client.completions.create.assert_called_once()

    def test_process_chat_format(self, mock_openai_client, sample_dataset):
        """Test processing chat format conversations."""

        # Create a mock tokenizer
        mock_tokenizer = Mock()

        def apply_template(conversation, add_generation_prompt=True):
            # Return deterministic token ids ensuring prefix property
            return list(range(10 if add_generation_prompt else 14))

        mock_tokenizer.apply_chat_template = Mock(side_effect=apply_template)
        mock_tokenizer.encode = Mock(side_effect=lambda text: list(range(len(text))))

        prompt: List[ChatMessage] = [{"role": "user", "content": "What is 2+2?"}]
        completion: List[ChatMessage] = [{"role": "assistant", "content": "4"}]
        # Minimal vLLM-style chat completion with tokens/logprobs
        token_entries = [
            Mock(logprob=-0.1, token="token_id:11"),
            Mock(logprob=-0.2, token="token_id:12"),
            Mock(logprob=-0.3, token="token_id:13"),
            Mock(logprob=-0.4, token="token_id:14"),
        ]
        mock_choice = Mock()
        mock_choice.logprobs = Mock()
        mock_choice.logprobs.content = token_entries
        mock_chat_completion = Mock()
        mock_chat_completion.choices = [mock_choice]
        state = {"responses": [mock_chat_completion]}

        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            completion_logprobs,
        ) = process_chat_format_vllm(
            prompt, completion, state, mock_tokenizer, mask_env_responses=False
        )

        assert isinstance(prompt_ids, list)
        assert isinstance(prompt_mask, list)
        assert isinstance(completion_ids, list)
        assert isinstance(completion_mask, list)
        assert isinstance(completion_logprobs, list)
        assert len(prompt_ids) == len(prompt_mask)
        assert len(completion_ids) == len(completion_mask)
        assert len(completion_ids) == len(completion_logprobs)
        assert all(m == 0 for m in prompt_mask)  # Prompt mask should be all 0s
        assert all(m == 1 for m in completion_mask)  # Completion mask should be all 1s

    def test_process_completion_format(self, mock_openai_client, sample_dataset):
        """Test processing completion format text."""

        # Create a mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(side_effect=lambda text: list(range(len(text))))

        prompt = "Complete this: 2+2="
        completion = "4"
        # Minimal vLLM-style completion covering entire completion string
        mock_choice = Mock()
        mock_choice.text = completion
        mock_choice.logprobs = Mock()
        mock_choice.logprobs.tokens = ["token_id:1"] * len(completion)
        mock_choice.logprobs.token_logprobs = [-0.1] * len(completion)
        mock_completion = Mock()
        mock_completion.choices = [mock_choice]
        state = {"responses": [mock_completion], "responses_start_idx": [0]}

        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            completion_logprobs,
        ) = process_completion_format_vllm(prompt, completion, state, mock_tokenizer)

        assert isinstance(prompt_ids, list)
        assert isinstance(prompt_mask, list)
        assert isinstance(completion_ids, list)
        assert isinstance(completion_mask, list)
        assert isinstance(completion_logprobs, list)
        assert len(prompt_ids) == len(prompt_mask)
        assert len(completion_ids) == len(completion_mask)
        assert len(completion_ids) == len(completion_logprobs)
        assert all(m == 0 for m in prompt_mask)
        assert all(m == 1 for m in completion_mask)

    def test_process_env_results_chat(self, mock_openai_client, sample_dataset):
        """Test processing environment results for chat format."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Create a mock tokenizer
        mock_tokenizer = Mock()

        # Track the conversation state
        def mock_apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        ):
            # Return token id list; size scales with number of messages
            return list(range(len(conversation) * 5))

        def mock_encode(text, **kwargs):
            return list(range(3))

        mock_tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)
        mock_tokenizer.encode = Mock(side_effect=mock_encode)

        prompts: List[Messages] = [[{"role": "user", "content": "Hello"}]]
        completions: List[Messages] = [[{"role": "assistant", "content": "Hi there!"}]]
        # Minimal vLLM-style chat completion mock for assistant turn
        token_entries = [
            Mock(logprob=-0.1, token="token_id:101"),
            Mock(logprob=-0.2, token="token_id:102"),
            Mock(logprob=-0.3, token="token_id:103"),
        ]
        mock_choice = Mock()
        mock_choice.logprobs = Mock()
        mock_choice.logprobs.content = token_entries
        mock_chat_completion = Mock()
        mock_chat_completion.choices = [mock_choice]
        states = [{"responses": [mock_chat_completion]}]
        rewards = [1.0]

        results = env.process_env_results_vllm(
            prompts, completions, states, rewards, mock_tokenizer
        )

        assert hasattr(results, "prompt_ids")
        assert hasattr(results, "prompt_mask")
        assert hasattr(results, "completion_ids")
        assert hasattr(results, "completion_mask")
        assert hasattr(results, "completion_logprobs")
        assert hasattr(results, "rewards")
        assert len(results.rewards) == 1
        assert results.rewards[0] == 1.0

    def test_process_env_results_with_truncation(
        self, mock_openai_client, sample_dataset
    ):
        """Test processing environment results with sequence length truncation."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Create a mock tokenizer
        mock_tokenizer = Mock()

        # Track the conversation state
        def mock_apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        ):
            # Return deterministic token ids based on number of messages
            # Ensures prefix property when conversation grows
            return list(range(len(conversation) * 5))

        def mock_encode(text, **kwargs):
            # Not used by chat path; keep for compatibility
            return list(range(3))

        mock_tokenizer.apply_chat_template = Mock(side_effect=mock_apply_chat_template)
        mock_tokenizer.encode = Mock(side_effect=mock_encode)

        prompts: List[Messages] = [[{"role": "user", "content": "Hello"}]]
        completions: List[Messages] = [[{"role": "assistant", "content": "Hi there!"}]]
        # Produce enough assistant tokens to force truncation
        token_entries = [
            Mock(logprob=-0.1, token="token_id:201"),
            Mock(logprob=-0.2, token="token_id:202"),
            Mock(logprob=-0.3, token="token_id:203"),
            Mock(logprob=-0.4, token="token_id:204"),
            Mock(logprob=-0.5, token="token_id:205"),
            Mock(logprob=-0.6, token="token_id:206"),
        ]
        mock_choice = Mock()
        mock_choice.logprobs = Mock()
        mock_choice.logprobs.content = token_entries
        mock_chat_completion = Mock()
        mock_chat_completion.choices = [mock_choice]
        states = [{"responses": [mock_chat_completion]}]
        rewards = [1.0]

        results = env.process_env_results_vllm(
            prompts,
            completions,
            states,
            rewards,
            mock_tokenizer,
            max_seq_len=8,  # Force truncation
            mask_truncated_completions=True,
        )

        # Check that total length respects max_seq_len
        total_len = len(results.prompt_ids[0]) + len(results.completion_ids[0])
        assert total_len <= 8
        # Check that truncated completion is masked
        assert all(m == 0 for m in results.completion_mask[0])

    def test_parse_chat_completion_logprobs(self, mock_openai_client, sample_dataset):
        """Test parsing logprobs from a vLLM chat completion."""

        # Create mock chat completion with logprobs
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].logprobs = Mock()
        mock_completion.choices[0].logprobs.content = [
            Mock(logprob=-0.5),
            Mock(logprob=-1.2),
            Mock(logprob=-0.3),
        ]

        logprobs = parse_chat_completion_logprobs(mock_completion)
        assert logprobs == [-0.5, -1.2, -0.3]

    def test_parse_chat_completion_tokens(self, mock_openai_client, sample_dataset):
        """Test parsing tokens from a vLLM chat completion."""

        # Create mock chat completion with tokens
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].logprobs = Mock()
        mock_completion.choices[0].logprobs.content = [
            Mock(token="id:1234"),
            Mock(token="id:5678"),
            Mock(token="id:9012"),
        ]

        tokens = parse_chat_completion_tokens(mock_completion)
        assert tokens == [1234, 5678, 9012]

    @pytest.mark.asyncio
    async def test_run_rollouts(self, mock_openai_client):
        """Test running multiple rollouts."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            eval_dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            parser=Parser(),
            rubric=Rubric(),
        )

        prompts: List[Messages] = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}],
        ]
        answers = ["response1", "response2"]
        tasks = ["default", "default"]
        infos = [{}, {}]

        # Mock the rollout method calls
        results = await env.run_rollouts(
            client=mock_openai_client,
            model="test-model",
            prompts=prompts,
            answers=answers,
            tasks=tasks,
            infos=infos,
            example_ids=list(range(len(prompts))),
        )

        assert len(results) == 2
        assert all(
            len(result) == 2 for result in results
        )  # Each result is (completion, state)

    @pytest.mark.asyncio
    async def test_a_generate_with_score_rollouts(
        self, mock_openai_client, sample_dataset
    ):
        """Test async generate with scoring enabled."""
        env = SimpleEnvironment(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the rubric scoring
        env.rubric.score_rollouts = AsyncMock(  # type: ignore[attr-defined]
            return_value=RolloutScores(reward=[1.0], metrics={})
        )

        inputs = {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "answer": ["Hi"],
            "example_id": [0],
        }

        results = await env.a_generate(
            inputs,
            client=mock_openai_client,
            model="test-model",
            score_rollouts=True,
            interleave_scoring=False,
        )

        assert hasattr(results, "completion")
        assert hasattr(results, "state")
        assert hasattr(results, "reward")
        assert results.reward == [1.0]

    def test_generate_sync_wrapper(self, mock_openai_client, sample_dataset):
        """Test synchronous generate wrapper."""
        env = SimpleEnvironment(
            dataset=sample_dataset,
            parser=Parser(),
            rubric=Rubric(),
        )

        # Mock the rubric scoring
        env.rubric.score_rollouts = AsyncMock(  # type: ignore[attr-defined]
            return_value=RolloutScores(reward=[1.0], metrics={})
        )

        inputs = GenerateInputs(
            prompt=[[{"role": "user", "content": "Hello"}]],
            answer=["Hi"],
            info=[{}],
            example_id=[0],
        )
        results = env.generate_sync(
            inputs,
            client=mock_openai_client,
            model="test-model",
            interleave_scoring=False,
        )

        assert hasattr(results, "completion")
        assert hasattr(results, "state")
        assert hasattr(results, "reward")

    def test_make_dataset(self, mock_openai_client, sample_dataset):
        """Test creating a dataset from evaluation results."""

        results = GenerateOutputs(
            prompt=[[{"role": "user", "content": "Hello"}]],
            completion=[[{"role": "assistant", "content": "Hi"}]],
            answer=["Hi"],
            reward=[1.0],
            task=["default"],
            state=[
                {
                    "custom_field": "value",
                    "timing": {
                        "generation_ms": 0.0,
                        "scoring_ms": 0.0,
                        "total_ms": 0.0,
                    },
                }
            ],
            info=[{}],
            example_id=[0],
            metrics={},
            metadata=_make_metadata(num_examples=1),
        )

        dataset = build_dataset(results)

        assert len(dataset) == 1
        assert "prompt" in dataset.column_names
        assert "completion" in dataset.column_names
        assert "answer" in dataset.column_names
        assert "reward" in dataset.column_names
        assert "task" in dataset.column_names
        assert "example_id" in dataset.column_names
        assert "custom_field" in dataset.column_names

    @pytest.mark.asyncio
    async def test_generate_state_preserves_references(self, mock_openai_client):
        """Test that generate creates state with preserved references instead of deep copying"""
        env = SimpleEnvironment(
            eval_dataset=Dataset.from_dict(
                {"question": ["test question"], "answer": ["test answer"]}
            ),
            parser=Parser(),
            rubric=Rubric(),
        )

        env.rubric.score_rollouts = AsyncMock(  # type: ignore[attr-defined]
            return_value=RolloutScores(reward=[1.0], metrics={})
        )

        inputs = {
            "prompt": [[{"role": "user", "content": "Hello"}]],
            "answer": ["Hi"],
            "info": [{"key": "value"}],
            "example_id": [0],
        }

        results = await env.generate(
            inputs,
            client=mock_openai_client,
            model="test-model",
            score_rollouts=True,
            interleave_scoring=False,
        )

        assert len(results.state) == 1
        state = results.state[0]

        assert state["prompt"] is results.prompt[0]
        assert state["completion"] is results.completion[0]
        assert state["answer"] is results.answer[0]
        assert state["info"] is results.info[0]
        assert state["example_id"] is results.example_id[0]
