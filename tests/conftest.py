"""Pytest configuration and fixtures for verifiers tests."""

from unittest.mock import AsyncMock, MagicMock

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
    ToolEnv,
    XMLParser,
    stop,
)


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


class MockAsyncOpenAI:
    """Mock AsyncOpenAI client that maps conversation inputs to outputs."""

    def __init__(self):
        self.chat_completions = {}  # Maps conversation history to responses
        self.text_completions = {}  # Maps prompts to responses
        self.default_chat_response = "This is a test response"
        self.default_text_response = "This is a test completion"
        self.base_url = "http://localhost/v1/"  # For testing URL parsing

        # Create mock structure
        self.chat = MagicMock()
        self.completions = MagicMock()
        self.chat.completions = MagicMock()

        # Set up async methods
        self.chat.completions.create = AsyncMock(
            side_effect=self._handle_chat_completion
        )
        self.completions.create = AsyncMock(side_effect=self._handle_text_completion)

    def add_chat_response(
        self, messages, response, finish_reason="stop", tool_calls=None
    ):
        """Add a mapped response for specific messages."""
        # Convert messages to a hashable key
        key = self._messages_to_key(messages)
        self.chat_completions[key] = {
            "content": response,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }

    def add_text_response(self, prompt, response, finish_reason="stop"):
        """Add a mapped response for specific prompt."""
        self.text_completions[prompt] = {
            "text": response,
            "finish_reason": finish_reason,
        }

    def set_default_responses(self, chat_response=None, text_response=None):
        """Set default responses when no mapping found."""
        if chat_response:
            self.default_chat_response = chat_response
        if text_response:
            self.default_text_response = text_response

    async def _handle_chat_completion(self, messages, **kwargs):
        """Handle chat completion requests."""
        key = self._messages_to_key(messages)

        if key in self.chat_completions:
            response_data = self.chat_completions[key]
        else:
            response_data = {
                "content": self.default_chat_response,
                "finish_reason": "stop",
                "tool_calls": None,
            }

        # Create mock response that mimics ChatCompletion
        from openai.types.chat.chat_completion import ChatCompletion, Choice
        from openai.types.chat.chat_completion_message import ChatCompletionMessage

        # Create a proper mock that will pass isinstance checks
        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock(spec=Choice)
        mock_message = MagicMock(spec=ChatCompletionMessage)

        # Set the attributes
        mock_message.content = response_data["content"]
        mock_message.role = "assistant"
        mock_message.tool_calls = response_data.get("tool_calls", None)
        mock_choice.message = mock_message
        mock_choice.finish_reason = response_data["finish_reason"]
        mock_choice.index = 0

        mock_response.choices = [mock_choice]
        mock_response.id = "test-id"
        mock_response.model = "test-model"
        mock_response.object = "chat.completion"

        return mock_response

    async def _handle_text_completion(self, prompt, **kwargs):
        """Handle text completion requests."""
        if prompt in self.text_completions:
            response_data = self.text_completions[prompt]
        else:
            response_data = {
                "text": self.default_text_response,
                "finish_reason": "stop",
            }

        # Create mock response that mimics Completion
        from openai.types.completion import Completion
        from openai.types.completion_choice import CompletionChoice

        # Create a proper mock that will pass isinstance checks
        mock_response = MagicMock(spec=Completion)
        mock_choice = MagicMock(spec=CompletionChoice)

        # Set the attributes
        mock_choice.text = response_data["text"]
        mock_choice.finish_reason = response_data["finish_reason"]
        mock_choice.index = 0

        mock_response.choices = [mock_choice]
        mock_response.id = "test-id"
        mock_response.model = "test-model"
        mock_response.object = "text_completion"

        return mock_response

    def _messages_to_key(self, messages):
        """Convert messages list to a hashable key."""
        # Create a simplified representation for hashing
        key_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            key_parts.append(f"{role}:{content}")
        return tuple(key_parts)


@pytest.fixture
def mock_openai_client():
    """Return a mocked AsyncOpenAI client with input-output mapping."""
    return MockAsyncOpenAI()


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
def mock_singleturn_env(mock_openai_client, sample_dataset):
    """Return a SingleTurnEnv with mocked client and dataset."""
    return SingleTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_dataset,
        system_prompt="You are a helpful assistant.",
        parser=Parser(),
        rubric=Rubric(),
    )


@pytest.fixture
def mock_singleturn_env_completion(mock_openai_client):
    """Return a SingleTurnEnv for completion format testing."""
    completion_dataset = Dataset.from_dict(
        {
            "prompt": ["Calculate 2+2:", "Name the capital of France:"],
            "answer": ["4", "Paris"],
        }
    )
    return SingleTurnEnv(
        client=mock_openai_client,
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
def mock_multiturn_env(mock_openai_client, sample_chat_dataset):
    """Return a MultiTurnEnv for basic testing."""
    return SimpleMultiTurnEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_chat_dataset,
        max_turns=3,
        completion_condition="answer",
        parser=Parser(),
        rubric=Rubric(),
    )


@pytest.fixture
def mock_multiturn_env_max_turns(mock_openai_client, sample_chat_dataset):
    """Return a MultiTurnEnv that tests max_turns limiting."""
    return SimpleMultiTurnEnv(
        client=mock_openai_client,
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
    raise ValueError("failure")


class BasicToolEnv(ToolEnv):
    def __init__(self, **kwargs):
        super().__init__(tools=[square_tool], **kwargs)


@pytest.fixture
def mock_tool_env(mock_openai_client, sample_chat_dataset):
    return BasicToolEnv(
        client=mock_openai_client,
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
def mock_stateful_tool_env(mock_openai_client, sample_chat_dataset):
    return ExampleStatefulToolEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_chat_dataset,
        parser=Parser(),
        rubric=Rubric(),
    )
