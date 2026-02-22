"""Pytest configuration and fixtures for verifiers tests."""

import logging
from pathlib import Path
from typing import Any, Callable

import pytest
from datasets import Dataset

from verifiers import (
    MaybeThinkParser,
    Messages,
    MultiTurnEnv,
    Parser,
    Rubric,
    SingleTurnEnv,
    State,
    StatefulToolEnv,
    ThinkParser,
    ToolCallError,
    ToolEnv,
    XMLParser,
    stop,
)
from verifiers.clients.client import Client
from verifiers.types import (
    GenerateMetadata,
    Info,
    Response,
    ResponseMessage,
    RolloutInput,
    RolloutOutput,
    RolloutTiming,
    SamplingArgs,
    Tool,
    ToolCall,
    TrajectoryStep,
)
from verifiers.utils.save_utils import state_to_output


@pytest.fixture
def basic_parser():
    """Return a basic Parser instance."""
    return Parser()


@pytest.fixture
def xml_parser():
    """Return an XMLParser instance with common fields."""
    return XMLParser(fields=["reasoning", "answer"], answer_field="answer")


@pytest.fixture
def xml_parser_with_alternatives():
    """Return an XMLParser instance with alternative field names."""
    return XMLParser(fields=["reasoning", ("code", "answer")], answer_field="answer")


@pytest.fixture
def maybe_think_parser():
    """Return a MaybeThinkParser instance."""
    return MaybeThinkParser()


@pytest.fixture
def think_parser():
    """Return a ThinkParser instance."""
    return ThinkParser()


@pytest.fixture
def think_parser_with_extractor():
    """Return a ThinkParser instance with custom extraction function."""

    def extract_boxed(text):
        """Simple boxed answer extractor for testing."""
        import re

        match = re.search(r"\\boxed\{([^}]+)\}", text)
        return match.group(1) if match else text

    return ThinkParser(extract_fn=extract_boxed)


# Async test fixtures for Environment testing


class MockClient(Client):
    """Mocked vf.Client with get_response() to return provider-agnostic vf.Response objects"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MockClient")
        self._client = None

        self._responses: dict[tuple, dict] = {}
        self.default_response = "This is a test response"

        # Call tracking
        self.call_count = 0
        self.last_call_kwargs: dict[str, Any] = {}

    def add_response(self, messages, response, finish_reason="stop", tool_calls=None):
        """Add a mapped response for specific messages."""
        key = self._messages_to_key(self._normalize_input(messages))
        self._responses[key] = {
            "content": response,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }

    def set_default_response(self, response):
        """Set default response when no mapping found."""
        self.default_response = response

    async def get_response(
        self,
        prompt,
        model,
        sampling_args,
        tools=None,
        **kwargs,
    ) -> Response:
        """Return a Response based on the prompt-to-response mapping."""
        self.call_count += 1
        self.last_call_kwargs = {
            "prompt": prompt,
            "model": model,
            "sampling_args": sampling_args,
            "tools": tools,
            **kwargs,
        }

        return self._make_response(prompt)

    def setup_client(self, config):
        return None

    async def to_native_tool(self, tool):
        pass

    async def to_native_prompt(self, messages):
        return [], {}

    async def get_native_response(
        self, prompt, model, sampling_args, tools=None, **kwargs
    ):
        pass

    async def raise_from_native_response(self, response):
        pass

    async def from_native_response(self, response):
        pass

    async def close(self) -> None:
        pass

    # -- Internal helpers --

    @staticmethod
    def _normalize_input(messages):
        """Normalize prompt to list-of-dicts form for keying."""
        if isinstance(messages, str):
            return [{"role": "text", "content": messages}]
        return messages

    def _messages_to_key(self, messages):
        """Convert messages list to a hashable key."""
        key_parts = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", "")
            key_parts.append(f"{role}:{content}")
        return tuple(key_parts)

    def _convert_tool_calls(self, raw_tool_calls) -> list[ToolCall] | None:
        """Convert OAI-style tool call objects to vf.ToolCall."""
        if not raw_tool_calls:
            return None
        result: list[ToolCall] = []
        for tc in raw_tool_calls:
            if hasattr(tc, "function"):
                result.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                )
            elif isinstance(tc, dict):
                func = tc.get("function", {})
                result.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=func.get("name", ""),
                        arguments=func.get("arguments", ""),
                    )
                )
        return result or None

    def _make_response(self, prompt) -> Response:
        key = self._messages_to_key(self._normalize_input(prompt))
        if key in self._responses:
            data = self._responses[key]
        else:
            data = {
                "content": self.default_response,
                "finish_reason": "stop",
                "tool_calls": None,
            }

        tool_calls = self._convert_tool_calls(data.get("tool_calls"))

        return Response(
            id="test-id",
            created=0,
            model="test-model",
            usage=None,
            message=ResponseMessage(
                content=data["content"],
                reasoning_content=None,
                finish_reason=data["finish_reason"],
                is_truncated=data["finish_reason"] == "length",
                tokens=None,
                tool_calls=tool_calls,
            ),
        )


@pytest.fixture
def mock_client():
    """Return a MockClient with input-output mapping."""
    return MockClient()


@pytest.fixture
def sample_dataset():
    """Return a sample dataset for testing."""
    return Dataset.from_dict(
        {
            "question": ["What is 2+2?", "What is the capital of France?"],
            "answer": ["4", "Paris"],
        }
    )


@pytest.fixture
def sample_chat_dataset():
    """Return a sample dataset with chat format."""
    return Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": "What is 2+2?"}],
                [{"role": "user", "content": "What is the capital of France?"}],
            ],
            "answer": ["4", "Paris"],
            "example_id": [0, 1],
        }
    )


@pytest.fixture
def mock_singleturn_env(mock_client, sample_dataset):
    """Return a SingleTurnEnv with mocked client and dataset."""
    return SingleTurnEnv(
        client=mock_client,
        model="test-model",
        dataset=sample_dataset,
        system_prompt="You are a helpful assistant.",
        parser=Parser(),
        rubric=Rubric(),
    )


@pytest.fixture
def mock_singleturn_env_completion(mock_client):
    """Return a SingleTurnEnv for completion format testing."""
    completion_dataset = Dataset.from_dict(
        {
            "prompt": ["Calculate 2+2:", "Name the capital of France:"],
            "answer": ["4", "Paris"],
        }
    )
    return SingleTurnEnv(
        client=mock_client,
        model="test-model",
        dataset=completion_dataset,
        message_type="completion",
        parser=Parser(),
        rubric=Rubric(),
    )


# MultiTurnEnv test fixtures


class SimpleMultiTurnEnv(MultiTurnEnv):
    """Simple concrete implementation of MultiTurnEnv for testing."""

    def __init__(self, completion_condition="answer", **kwargs):
        super().__init__(**kwargs)
        self.completion_condition = (
            completion_condition  # "answer", "max_turns", "error"
        )
        self.env_response_count = 0

    @stop
    async def done_condition(self, state: State) -> bool:
        """Complete when assistant says 'DONE'."""
        if self.completion_condition == "answer":
            if state["trajectory"]:
                last_completion = state["trajectory"][-1]["completion"]
                if isinstance(last_completion, list) and last_completion:
                    return "DONE" in str(last_completion[-1].get("content", ""))
                elif isinstance(last_completion, str):
                    return "DONE" in last_completion
        return False

    @stop
    async def error_condition(self, state: State) -> bool:
        """Complete on any error."""
        if self.completion_condition == "error":
            if state["trajectory"]:
                last_completion = state["trajectory"][-1]["completion"]
                if isinstance(last_completion, list) and last_completion:
                    return str(last_completion[-1].get("content", "")).startswith(
                        "[ERROR]"
                    )
                elif isinstance(last_completion, str):
                    return last_completion.startswith("[ERROR]")
        return False

    async def env_response(self, messages, state, **kwargs) -> Messages:
        """Simple environment response for testing."""
        self.env_response_count += 1

        if self.completion_condition == "answer":
            # Encourage completion after a few turns
            if self.env_response_count >= 2:
                return [{"role": "user", "content": "Please finish with DONE"}]
            else:
                return [
                    {
                        "role": "user",
                        "content": f"Continue (turn {self.env_response_count})",
                    }
                ]
        else:
            return [
                {
                    "role": "user",
                    "content": f"Environment response {self.env_response_count}",
                }
            ]


@pytest.fixture
def mock_multiturn_env(mock_client, sample_chat_dataset):
    """Return a MultiTurnEnv for basic testing."""
    return SimpleMultiTurnEnv(
        client=mock_client,
        model="test-model",
        dataset=sample_chat_dataset,
        max_turns=3,
        completion_condition="answer",
        parser=Parser(),
        rubric=Rubric(),
    )


@pytest.fixture
def mock_multiturn_env_max_turns(mock_client, sample_chat_dataset):
    """Return a MultiTurnEnv that tests max_turns limiting."""
    return SimpleMultiTurnEnv(
        client=mock_client,
        model="test-model",
        dataset=sample_chat_dataset,
        max_turns=2,
        completion_condition="max_turns",  # Never complete naturally
        parser=Parser(),
        rubric=Rubric(),
    )


def square_tool(x: int) -> int:
    return x * x


def faulty_tool() -> None:
    cause = ValueError("failure")
    raise ToolCallError from cause


class BasicToolEnv(ToolEnv):
    def __init__(self, **kwargs):
        super().__init__(tools=[square_tool], **kwargs)


@pytest.fixture
def mock_tool_env(mock_client, sample_chat_dataset):
    return BasicToolEnv(
        client=mock_client,
        model="test-model",
        dataset=sample_chat_dataset,
        parser=Parser(),
        rubric=Rubric(),
    )


def offset_tool(x: int, offset: int) -> int:
    return x + offset


def secret_tool(x: int, secret: int) -> int:
    return x + secret


class ExampleStatefulToolEnv(StatefulToolEnv):
    def __init__(self, **kwargs):
        super().__init__(tools=[offset_tool], **kwargs)

    async def setup_state(self, state, **kwargs):
        state = await super().setup_state(state, **kwargs)
        state["offset"] = 3
        state["update_calls"] = 0
        return state

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
        state["update_calls"] += 1
        updated_args = {**tool_args, "offset": state["offset"]}
        state["last_tool_args"] = updated_args.copy()
        return updated_args


@pytest.fixture
def mock_stateful_tool_env(mock_client, sample_chat_dataset):
    return ExampleStatefulToolEnv(
        client=mock_client,
        model="test-model",
        dataset=sample_chat_dataset,
        parser=Parser(),
        rubric=Rubric(),
    )


DEFAULT_PROMPT: Messages = [{"role": "user", "content": "What is 2+2?"}]
DEFAULT_COMPLETION: Messages = [{"role": "assistant", "content": "4"}]


@pytest.fixture
def make_input() -> Callable[..., RolloutInput]:
    """Fixture to make RolloutInput objects for testing."""

    def _make_input(
        example_id: int = 0,
        task: str = "default",
        prompt: Messages = DEFAULT_PROMPT,
        info: Info = {},
        answer: str = "4",
    ) -> RolloutInput:
        return RolloutInput(
            example_id=example_id, task=task, prompt=prompt, answer=answer, info=info
        )

    return _make_input


@pytest.fixture
def make_state() -> Callable[..., State]:
    """Fixture to make State objects for testing."""

    def _make_state(
        example_id: int = 0,
        task: str = "default",
        prompt: Messages = DEFAULT_PROMPT,
        answer: str = "4",
        info: Info = {},
        completion: Messages = DEFAULT_COMPLETION,
        reward: float = 0.0,
        metrics: dict[str, float] = {"accuracy": 0.0},
        is_completed: bool = True,
        is_truncated: bool = False,
        stop_condition: str | None = "max_turns_reached",
        tool_defs: list[Tool] | None = None,
        trajectory: list[TrajectoryStep] = [],
        timing=RolloutTiming(
            generation_ms=0.0,
            scoring_ms=0.0,
            total_ms=0.0,
        ),
        foo: str = "bar",  # custom field
        **kwargs,
    ) -> State:
        return State(
            example_id=example_id,
            task=task,
            prompt=prompt,
            answer=answer,
            info=info,
            completion=completion,
            reward=reward,
            metrics=metrics,
            is_completed=is_completed,
            is_truncated=is_truncated,
            stop_condition=stop_condition,
            tool_defs=tool_defs,
            trajectory=trajectory,
            timing=timing,
            error=None,
            foo=foo,
            **kwargs,
        )

    return _make_state


@pytest.fixture
def make_output(make_state) -> Callable[..., RolloutOutput]:
    """Fixture to make RolloutOutput objects for testing.

    This creates a State first, then converts it to a RolloutOutput using
    state_to_output(). This ensures the output matches the serialized format
    used in GenerateOutputs.
    """

    def _make_output(
        state_columns: list[str] = ["foo"],
        **kwargs,
    ) -> RolloutOutput:
        state = make_state(**kwargs)
        return state_to_output(state, state_columns)

    return _make_output


@pytest.fixture
def make_metadata() -> Callable[..., GenerateMetadata]:
    """Fixture to make GenerateMetadata objects for testing."""

    def _make_metadata(
        env_id: str = "test-env",
        env_args: dict = {},
        model: str = "test-model",
        base_url: str = "http://localhost:8000/v1",
        num_examples: int = 1,
        rollouts_per_example: int = 1,
        sampling_args: SamplingArgs = {},
        date: str = "1970-01-01",
        time_ms: float = 0.0,
        avg_reward: float = 0.0,
        avg_metrics: dict[str, float] = {},
        pass_at_k: dict[str, float] = {},
        pass_all_k: dict[str, float] = {},
        pass_threshold: float = 0.5,
        usage: dict[str, float] | None = None,
        version_info: dict | None = None,
        state_columns: list[str] = ["foo"],
        path_to_save: Path = Path("test.jsonl"),
        tools: list[Tool] | None = None,
    ) -> GenerateMetadata:
        if version_info is None:
            version_info = {
                "vf_version": "0.0.0-test",
                "vf_commit": None,
                "env_version": None,
                "env_commit": None,
            }
        return GenerateMetadata(
            env_id=env_id,
            env_args=env_args,
            model=model,
            base_url=base_url,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            sampling_args=sampling_args,
            date=date,
            time_ms=time_ms,
            avg_reward=avg_reward,
            avg_metrics=avg_metrics,
            pass_at_k=pass_at_k,
            pass_all_k=pass_all_k,
            pass_threshold=pass_threshold,
            usage=usage,
            version_info=version_info,
            state_columns=state_columns,
            path_to_save=path_to_save,
            tools=tools,
        )

    return _make_metadata
