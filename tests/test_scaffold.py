"""Tests for the Scaffold classes."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import BadRequestError
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

import verifiers as vf
from verifiers.scaffolds import Scaffold, ScaffoldResult, ToolCall, ToolScaffold
from verifiers.types import RolloutInput


# ============================================================================
# Test Fixtures
# ============================================================================


def _make_chat_response(
    content: str = "Test response",
    tool_calls: list | None = None,
    finish_reason: str = "stop",
) -> ChatCompletion:
    """Helper to create a mock ChatCompletion response."""
    mock_message = MagicMock(spec=ChatCompletionMessage)
    mock_message.content = content
    mock_message.role = "assistant"
    mock_message.tool_calls = tool_calls
    mock_message.model_dump = lambda: {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (tool_calls or [])
        ] if tool_calls else None,
    }

    mock_choice = MagicMock(spec=Choice)
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason
    mock_choice.index = 0

    mock_response = MagicMock(spec=ChatCompletion)
    mock_response.choices = [mock_choice]
    mock_response.id = "test-id"
    mock_response.model = "test-model"
    mock_response.object = "chat.completion"

    return mock_response


def _make_tool_call(name: str, arguments: dict, tool_call_id: str = "call_0"):
    """Helper to create a tool call object."""
    return ChatCompletionMessageToolCall(
        id=tool_call_id,
        type="function",
        function=Function(name=name, arguments=json.dumps(arguments)),
    )


def _make_bad_request_error(message: str) -> BadRequestError:
    """Helper to create a BadRequestError."""
    mock_response = MagicMock()
    mock_response.text = message
    mock_response.status_code = 400
    mock_response.headers = {}
    return BadRequestError(message, response=mock_response, body=None)


@pytest.fixture
def mock_client():
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.completions = MagicMock()
    client.completions.create = AsyncMock()
    client.base_url = "http://localhost/v1/"
    return client


# ============================================================================
# Base Scaffold Tests
# ============================================================================


class TestScaffold:
    """Tests for the base Scaffold class."""

    @pytest.mark.asyncio
    async def test_basic_generate(self, mock_client):
        """Test basic generate call returns ScaffoldResult."""
        mock_client.chat.completions.create.return_value = _make_chat_response("Hello!")

        scaffold = Scaffold(
            client=mock_client,
            model="test-model",
            sampling_args={"temperature": 0},
        )

        messages = [{"role": "user", "content": "Hi"}]
        result = await scaffold.generate(messages, state=None)

        assert isinstance(result, ScaffoldResult)
        assert result.tool_calls_made == 0
        assert not result.has_pending_tool_calls
        # Messages should include original + assistant response
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "user"
        assert result.messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_sampling_args_normalization(self, mock_client):
        """Test that max_tokens is converted to max_completion_tokens for chat."""
        mock_client.chat.completions.create.return_value = _make_chat_response()

        scaffold = Scaffold(
            client=mock_client,
            model="test-model",
            sampling_args={"max_tokens": 100, "temperature": 0},
        )

        messages = [{"role": "user", "content": "Hi"}]
        await scaffold.generate(messages, state=None)

        # Check that the API was called with max_completion_tokens, not max_tokens
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "max_completion_tokens" in call_kwargs
        assert "max_tokens" not in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 100

    @pytest.mark.asyncio
    async def test_overlong_prompt_error(self, mock_client):
        """Test that context length errors raise OverlongPromptError."""
        error = _make_bad_request_error(
            "This model's maximum context length is 4096 tokens"
        )
        mock_client.chat.completions.create.side_effect = error

        scaffold = Scaffold(
            client=mock_client,
            model="test-model",
            sampling_args={},
        )

        messages = [{"role": "user", "content": "Hi"}]
        with pytest.raises(vf.OverlongPromptError):
            await scaffold.generate(messages, state=None)

    @pytest.mark.asyncio
    async def test_other_bad_request_becomes_model_error(self, mock_client):
        """Test that non-context-length BadRequestErrors become ModelError."""
        error = _make_bad_request_error("Invalid request format")
        mock_client.chat.completions.create.side_effect = error

        scaffold = Scaffold(
            client=mock_client,
            model="test-model",
            sampling_args={},
        )

        messages = [{"role": "user", "content": "Hi"}]
        with pytest.raises(vf.ModelError):
            await scaffold.generate(messages, state=None)

    @pytest.mark.asyncio
    async def test_generic_exception_becomes_model_error(self, mock_client):
        """Test that generic exceptions become ModelError."""
        mock_client.chat.completions.create.side_effect = RuntimeError("Connection failed")

        scaffold = Scaffold(
            client=mock_client,
            model="test-model",
            sampling_args={},
        )

        messages = [{"role": "user", "content": "Hi"}]
        with pytest.raises(vf.ModelError):
            await scaffold.generate(messages, state=None)

    @pytest.mark.asyncio
    async def test_tools_from_state_fallback(self, mock_client):
        """Test that scaffold uses tools from state if not set on scaffold."""
        mock_client.chat.completions.create.return_value = _make_chat_response()

        scaffold = Scaffold(
            client=mock_client,
            model="test-model",
            sampling_args={},
            oai_tools=None,  # No tools on scaffold
        )

        state = {
            "oai_tools": [{"type": "function", "function": {"name": "test_tool"}}]
        }
        messages = [{"role": "user", "content": "Hi"}]
        await scaffold.generate(messages, state=state)

        # Check that tools from state were passed to API
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("tools") == state["oai_tools"]

    @pytest.mark.asyncio
    async def test_empty_response_raises_error(self, mock_client):
        """Test that empty response raises EmptyModelResponseError."""
        mock_response = MagicMock(spec=ChatCompletion)
        mock_response.choices = None
        mock_client.chat.completions.create.return_value = mock_response

        scaffold = Scaffold(
            client=mock_client,
            model="test-model",
            sampling_args={},
        )

        messages = [{"role": "user", "content": "Hi"}]
        with pytest.raises(vf.EmptyModelResponseError):
            await scaffold.generate(messages, state=None)


# ============================================================================
# ToolScaffold Tests
# ============================================================================


def calculator_tool(expression: str) -> str:
    """A simple calculator tool for testing."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def greeting_tool(name: str) -> str:
    """A simple greeting tool for testing."""
    return f"Hello, {name}!"


class TestToolScaffold:
    """Tests for the ToolScaffold class."""

    @pytest.mark.asyncio
    async def test_no_tool_calls(self, mock_client):
        """Test that scaffold returns when model doesn't call tools."""
        mock_client.chat.completions.create.return_value = _make_chat_response(
            "The answer is 42"
        )

        scaffold = ToolScaffold(
            client=mock_client,
            model="test-model",
            tools=[calculator_tool],
            sampling_args={},
        )

        messages = [{"role": "user", "content": "What is 6*7?"}]
        result = await scaffold.generate(messages, state=None)

        assert result.tool_calls_made == 0
        assert not result.has_pending_tool_calls
        assert mock_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_internal_tool_execution(self, mock_client):
        """Test that internal tools are executed and loop continues."""
        tool_call = _make_tool_call("calculator_tool", {"expression": "6*7"})

        # First call: model calls tool
        response1 = _make_chat_response("Let me calculate", tool_calls=[tool_call])
        # Second call: model responds with answer
        response2 = _make_chat_response("The answer is 42")

        mock_client.chat.completions.create.side_effect = [response1, response2]

        scaffold = ToolScaffold(
            client=mock_client,
            model="test-model",
            tools=[calculator_tool],
            sampling_args={},
        )

        messages = [{"role": "user", "content": "What is 6*7?"}]
        result = await scaffold.generate(messages, state=None)

        assert result.tool_calls_made == 1
        assert not result.has_pending_tool_calls
        assert mock_client.chat.completions.create.call_count == 2

        # Check that tool result is in messages
        tool_messages = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["content"] == "42"

    @pytest.mark.asyncio
    async def test_yields_on_external_tool(self, mock_client):
        """Test that scaffold yields when model calls an unknown (env) tool."""
        # Model calls a tool that's not in our tool_map
        tool_call = _make_tool_call("submit_answer", {"answer": "42"})
        response = _make_chat_response("Submitting answer", tool_calls=[tool_call])

        mock_client.chat.completions.create.return_value = response

        scaffold = ToolScaffold(
            client=mock_client,
            model="test-model",
            tools=[calculator_tool],  # submit_answer is NOT here
            sampling_args={},
        )

        messages = [{"role": "user", "content": "Submit 42"}]
        result = await scaffold.generate(messages, state=None)

        # Should yield with pending tool calls
        assert result.has_pending_tool_calls
        assert len(result.pending_tool_calls) == 1
        assert result.pending_tool_calls[0].name == "submit_answer"
        assert result.pending_tool_calls[0].arguments == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_mixed_internal_and_external_tools(self, mock_client):
        """Test that internal tools are executed before yielding on external."""
        internal_call = _make_tool_call("calculator_tool", {"expression": "6*7"}, "call_1")
        external_call = _make_tool_call("submit_answer", {"answer": "42"}, "call_2")

        response = _make_chat_response(
            "Calculating and submitting",
            tool_calls=[internal_call, external_call]
        )
        mock_client.chat.completions.create.return_value = response

        scaffold = ToolScaffold(
            client=mock_client,
            model="test-model",
            tools=[calculator_tool],
            sampling_args={},
        )

        messages = [{"role": "user", "content": "Calculate 6*7 and submit"}]
        result = await scaffold.generate(messages, state=None)

        # Internal tool should have been executed
        assert result.tool_calls_made == 1

        # Should yield with external tool
        assert result.has_pending_tool_calls
        assert len(result.pending_tool_calls) == 1
        assert result.pending_tool_calls[0].name == "submit_answer"

        # Check that internal tool result is in messages
        tool_messages = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["content"] == "42"

    @pytest.mark.asyncio
    async def test_max_tool_turns_limit(self, mock_client):
        """Test that scaffold stops after max_tool_turns."""
        tool_call = _make_tool_call("calculator_tool", {"expression": "1+1"})
        response_with_tool = _make_chat_response("Calculating", tool_calls=[tool_call])

        # Always return a response with tool call
        mock_client.chat.completions.create.return_value = response_with_tool

        scaffold = ToolScaffold(
            client=mock_client,
            model="test-model",
            tools=[calculator_tool],
            sampling_args={},
            max_tool_turns=3,
        )

        messages = [{"role": "user", "content": "Keep calculating"}]
        result = await scaffold.generate(messages, state=None)

        # Should have made exactly max_tool_turns API calls
        assert mock_client.chat.completions.create.call_count == 3
        assert result.tool_calls_made == 3
        assert result.metadata.get("max_turns_hit") is True

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_client):
        """Test that tool errors are returned as error messages."""

        def failing_tool(x: int) -> str:
            raise ValueError("Tool failed!")

        tool_call = _make_tool_call("failing_tool", {"x": 1})
        response1 = _make_chat_response("Calling tool", tool_calls=[tool_call])
        response2 = _make_chat_response("Tool failed, sorry")

        mock_client.chat.completions.create.side_effect = [response1, response2]

        scaffold = ToolScaffold(
            client=mock_client,
            model="test-model",
            tools=[failing_tool],
            sampling_args={},
        )

        messages = [{"role": "user", "content": "Call the tool"}]
        result = await scaffold.generate(messages, state=None)

        # Tool should have been called (and failed)
        assert result.tool_calls_made == 1

        # Error message should be in tool response
        tool_messages = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert "Error" in tool_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_multiple_tools(self, mock_client):
        """Test scaffold with multiple internal tools."""
        tool_call1 = _make_tool_call("calculator_tool", {"expression": "2+2"}, "call_1")
        tool_call2 = _make_tool_call("greeting_tool", {"name": "World"}, "call_2")

        response1 = _make_chat_response(
            "Using both tools",
            tool_calls=[tool_call1, tool_call2]
        )
        response2 = _make_chat_response("Done!")

        mock_client.chat.completions.create.side_effect = [response1, response2]

        scaffold = ToolScaffold(
            client=mock_client,
            model="test-model",
            tools=[calculator_tool, greeting_tool],
            sampling_args={},
        )

        messages = [{"role": "user", "content": "Calculate and greet"}]
        result = await scaffold.generate(messages, state=None)

        assert result.tool_calls_made == 2

        tool_messages = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 2

        contents = [m["content"] for m in tool_messages]
        assert "4" in contents
        assert "Hello, World!" in contents


# ============================================================================
# ScaffoldResult Tests
# ============================================================================


class TestScaffoldResult:
    """Tests for the ScaffoldResult dataclass."""

    def test_has_pending_tool_calls_true(self):
        """Test has_pending_tool_calls returns True when there are pending calls."""
        result = ScaffoldResult(
            response=MagicMock(),
            messages=[],
            tool_calls_made=0,
            pending_tool_calls=[ToolCall(id="1", name="test", arguments={})],
        )
        assert result.has_pending_tool_calls is True

    def test_has_pending_tool_calls_false_none(self):
        """Test has_pending_tool_calls returns False when None."""
        result = ScaffoldResult(
            response=MagicMock(),
            messages=[],
            tool_calls_made=0,
            pending_tool_calls=None,
        )
        assert result.has_pending_tool_calls is False

    def test_has_pending_tool_calls_false_empty(self):
        """Test has_pending_tool_calls returns False when empty list."""
        result = ScaffoldResult(
            response=MagicMock(),
            messages=[],
            tool_calls_made=0,
            pending_tool_calls=[],
        )
        assert result.has_pending_tool_calls is False


# ============================================================================
# Integration Tests: Scaffold + Environment
# ============================================================================


class TestScaffoldEnvironmentIntegration:
    """Tests for scaffold integration with environments."""

    @pytest.mark.asyncio
    async def test_singleturn_env_with_scaffold(self, mock_openai_client, sample_dataset):
        """Test that SingleTurnEnv works with scaffold."""
        mock_openai_client.set_default_responses(chat_response="The answer is 4")

        env = vf.SingleTurnEnv(
            dataset=sample_dataset,
            system_prompt="You are helpful.",
        )

        scaffold = Scaffold(
            client=mock_openai_client,
            model="test-model",
            sampling_args={},
        )

        # Get formatted dataset
        ds = env.get_dataset()
        example = ds[0]

        state = await env.rollout(
            RolloutInput(**example),
            scaffold,
        )

        assert state["completion"] is not None
        assert len(state["trajectory"]) == 1

    @pytest.mark.asyncio
    async def test_tool_scaffold_messages_in_completion(
        self, mock_openai_client, sample_dataset
    ):
        """Test that ToolScaffold messages are used in completion."""
        # Setup mock to return tool call then final response
        tool_call = _make_tool_call("calculator_tool", {"expression": "2+2"})

        # Create proper mock responses
        response1 = _make_chat_response("Calculating", tool_calls=[tool_call])
        response2 = _make_chat_response("The answer is 4")

        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=[response1, response2]
        )

        env = vf.SingleTurnEnv(
            dataset=sample_dataset,
            system_prompt="You are helpful.",
        )

        scaffold = ToolScaffold(
            client=mock_openai_client,
            model="test-model",
            tools=[calculator_tool],
            sampling_args={},
        )

        ds = env.get_dataset()
        example = ds[0]

        state = await env.rollout(
            RolloutInput(**example),
            scaffold,
        )

        # Completion should include tool calls from scaffold_messages
        completion = state["completion"]
        assert state["scaffold_tool_calls"] == 1

        # Should have assistant, tool, assistant messages
        roles = [m.get("role") for m in completion]
        assert "tool" in roles

    @pytest.mark.asyncio
    async def test_scaffold_stored_in_state(self, mock_openai_client, sample_dataset):
        """Test that scaffold is stored in state for concurrency safety."""
        mock_openai_client.set_default_responses(chat_response="Test")

        env = vf.SingleTurnEnv(
            dataset=sample_dataset,
        )

        scaffold = Scaffold(
            client=mock_openai_client,
            model="test-model",
            sampling_args={},
        )

        ds = env.get_dataset()
        example = ds[0]

        state = await env.init_state(RolloutInput(**example), scaffold)

        # Scaffold should be stored in state
        assert state["scaffold"] is scaffold
        # Backwards compat: client/model should also be stored
        assert state["client"] is mock_openai_client
        assert state["model"] == "test-model"


# ============================================================================
# Scaffold Lifecycle Tests
# ============================================================================


class TestScaffoldLifecycle:
    """Tests for scaffold lifecycle management by environments."""

    @pytest.mark.asyncio
    async def test_setup_called_on_init_state(self, mock_openai_client, sample_dataset):
        """Test that scaffold.setup() is called when env.init_state() is called."""
        env = vf.SingleTurnEnv(dataset=sample_dataset)

        scaffold = Scaffold(
            client=mock_openai_client,
            model="test-model",
            sampling_args={},
        )

        assert not scaffold._setup_complete

        ds = env.get_dataset()
        example = ds[0]
        await env.init_state(RolloutInput(**example), scaffold)

        assert scaffold._setup_complete

    @pytest.mark.asyncio
    async def test_scaffold_registered_with_env(self, mock_openai_client, sample_dataset):
        """Test that scaffold is registered in env._scaffolds."""
        env = vf.SingleTurnEnv(dataset=sample_dataset)

        scaffold = Scaffold(
            client=mock_openai_client,
            model="test-model",
            sampling_args={},
        )

        assert len(env._scaffolds) == 0

        ds = env.get_dataset()
        example = ds[0]
        await env.init_state(RolloutInput(**example), scaffold)

        assert scaffold in env._scaffolds
        assert len(env._scaffolds) == 1

    @pytest.mark.asyncio
    async def test_scaffold_not_duplicated_on_multiple_init(
        self, mock_openai_client, sample_dataset
    ):
        """Test that same scaffold isn't added twice to env._scaffolds."""
        env = vf.SingleTurnEnv(dataset=sample_dataset)

        scaffold = Scaffold(
            client=mock_openai_client,
            model="test-model",
            sampling_args={},
        )

        ds = env.get_dataset()
        example = ds[0]

        # Call init_state multiple times with same scaffold
        await env.init_state(RolloutInput(**example), scaffold)
        await env.init_state(RolloutInput(**example), scaffold)
        await env.init_state(RolloutInput(**example), scaffold)

        # Should only be registered once
        assert len(env._scaffolds) == 1

    @pytest.mark.asyncio
    async def test_teardown_called_on_env_teardown(
        self, mock_openai_client, sample_dataset
    ):
        """Test that scaffold.teardown() is called when env._teardown() is called."""
        env = vf.SingleTurnEnv(dataset=sample_dataset)

        scaffold = Scaffold(
            client=mock_openai_client,
            model="test-model",
            sampling_args={},
        )

        ds = env.get_dataset()
        example = ds[0]
        await env.init_state(RolloutInput(**example), scaffold)

        assert scaffold._setup_complete

        # Trigger env teardown
        await env._teardown()

        assert not scaffold._setup_complete
        assert len(env._scaffolds) == 0

    @pytest.mark.asyncio
    async def test_multiple_scaffolds_all_torn_down(
        self, mock_openai_client, sample_dataset
    ):
        """Test that all scaffolds are torn down when env tears down."""
        env = vf.SingleTurnEnv(dataset=sample_dataset)

        scaffold1 = Scaffold(
            client=mock_openai_client,
            model="test-model",
            sampling_args={},
        )
        scaffold2 = Scaffold(
            client=mock_openai_client,
            model="test-model-2",
            sampling_args={},
        )

        ds = env.get_dataset()
        example = ds[0]

        await env.init_state(RolloutInput(**example), scaffold1)
        await env.init_state(RolloutInput(**example), scaffold2)

        assert len(env._scaffolds) == 2
        assert scaffold1._setup_complete
        assert scaffold2._setup_complete

        await env._teardown()

        assert not scaffold1._setup_complete
        assert not scaffold2._setup_complete
        assert len(env._scaffolds) == 0

    @pytest.mark.asyncio
    async def test_setup_is_idempotent(self, mock_openai_client):
        """Test that scaffold.setup() can be called multiple times safely."""
        scaffold = Scaffold(
            client=mock_openai_client,
            model="test-model",
            sampling_args={},
        )

        await scaffold.setup()
        assert scaffold._setup_complete

        # Call again - should not raise
        await scaffold.setup()
        assert scaffold._setup_complete

    @pytest.mark.asyncio
    async def test_teardown_is_idempotent(self, mock_openai_client):
        """Test that scaffold.teardown() can be called multiple times safely."""
        scaffold = Scaffold(
            client=mock_openai_client,
            model="test-model",
            sampling_args={},
        )

        await scaffold.setup()
        assert scaffold._setup_complete

        await scaffold.teardown()
        assert not scaffold._setup_complete

        # Call again - should not raise
        await scaffold.teardown()
        assert not scaffold._setup_complete

    @pytest.mark.asyncio
    async def test_teardown_without_setup_is_safe(self, mock_openai_client):
        """Test that calling teardown without setup is safe."""
        scaffold = Scaffold(
            client=mock_openai_client,
            model="test-model",
            sampling_args={},
        )

        assert not scaffold._setup_complete

        # Should not raise
        await scaffold.teardown()
        assert not scaffold._setup_complete
