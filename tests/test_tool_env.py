"""Tests for the ToolEnv class."""

import json
from json import JSONDecodeError

import pytest
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

import verifiers as vf
from tests.conftest import faulty_tool, offset_tool, square_tool
from verifiers.utils.tool_utils import is_valid_tool_content_parts


class TestIsValidToolContentParts:
    def test_valid_text_content_part(self):
        """Valid list with text content parts."""
        content = [{"type": "text", "text": "Hello world"}]
        assert is_valid_tool_content_parts(content) is True

    def test_valid_image_url_content_part(self):
        """Valid list with image_url content parts."""
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
        ]
        assert is_valid_tool_content_parts(content) is True

    def test_valid_mixed_content_parts(self):
        """Valid list with mixed text and image_url content parts."""
        content = [
            {"type": "text", "text": "Here's the screenshot"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]
        assert is_valid_tool_content_parts(content) is True

    def test_empty_list_is_valid(self):
        """Empty list is valid (no invalid parts)."""
        assert is_valid_tool_content_parts([]) is True

    def test_invalid_type_value(self):
        """Content part with invalid type value should fail."""
        content = [{"type": "invalid_type", "data": "some data"}]
        assert is_valid_tool_content_parts(content) is False

    def test_missing_type_key(self):
        """Content part without type key should fail."""
        content = [{"text": "Hello world"}]
        assert is_valid_tool_content_parts(content) is False

    def test_non_dict_item_in_list(self):
        """Non-dict item in list should fail."""
        content = ["just a string", {"type": "text", "text": "hello"}]
        assert is_valid_tool_content_parts(content) is False

    def test_non_list_input(self):
        """Non-list input should fail."""
        assert is_valid_tool_content_parts("just a string") is False
        assert is_valid_tool_content_parts({"type": "text", "text": "hi"}) is False
        assert is_valid_tool_content_parts(42) is False
        assert is_valid_tool_content_parts(None) is False

    def test_list_of_primitives(self):
        """List of primitives should fail (not valid content parts)."""
        assert is_valid_tool_content_parts([1, 2, 3]) is False
        assert is_valid_tool_content_parts(["a", "b", "c"]) is False


def _build_tool_call(name: str, arguments: dict, tool_call_id: str = "call_0"):
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )

    return ChatCompletionMessageToolCall(
        id=tool_call_id,
        type="function",
        function=Function(name=name, arguments=json.dumps(arguments)),
    )


class TestToolEnv:
    @pytest.mark.asyncio
    async def test_tool_env_calls_tool(
        self, mock_tool_env, mock_openai_client, make_input
    ):
        tool_call = _build_tool_call("square_tool", {"x": 4})
        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
        }
        user_message = ChatCompletionUserMessageParam(content="Square 4", role="user")

        mock_openai_client.add_chat_response(
            messages=[user_message],
            response="Using tool",
            tool_calls=[tool_call],
        )
        mock_openai_client.add_chat_response(
            messages=[
                user_message,
                assistant_message,
                {"role": "tool", "content": "16", "tool_call_id": "call_0"},
            ],
            response="Done",
        )

        state = await mock_tool_env.rollout(
            input=make_input(prompt=[user_message], answer="", task=""),
            client=mock_openai_client,
            model="test-model",
        )
        completion = state["completion"]

        tool_messages = [m for m in completion if m.get("role") == "tool"]
        assert tool_messages and tool_messages[0]["content"] == "16"
        assert (
            state["trajectory"][0]["response"].choices[0].message.tool_calls is not None
        )

    @pytest.mark.asyncio
    async def test_tool_env_completion_without_tool_calls(
        self, mock_tool_env, mock_openai_client, make_input
    ):
        mock_openai_client.add_chat_response(
            messages=[{"role": "user", "content": "Hello"}],
            response="Hi",
        )

        state = await mock_tool_env.rollout(
            input=make_input(
                prompt=[{"role": "user", "content": "Hello"}], answer="", task=""
            ),
            client=mock_openai_client,
            model="test-model",
        )
        completion = state["completion"]

        assert len(state["trajectory"]) == 1
        assert completion[-1]["role"] == "assistant"
        assert completion[-1]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_tool_env_tool_invalid_json_arguments(
        self, mock_openai_client, sample_chat_dataset, make_input
    ):
        """Test that ToolEnv stops rollout when tool call is not JSON-parsable."""

        class TestToolEnv(vf.ToolEnv):
            def __init__(self, **kwargs):
                super().__init__(
                    tools=[square_tool], stop_errors=[JSONDecodeError], **kwargs
                )

        env = TestToolEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
            parser=vf.Parser(),
            rubric=vf.Rubric(),
        )

        # Create a tool call with invalid JSON arguments
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
            Function,
        )

        tool_call_with_invalid_json_arguments = ChatCompletionMessageToolCall(
            id="call_0",
            type="function",
            function=Function(
                name="square_tool",
                arguments='{"x": invalid json}',  # Invalid JSON
            ),
        )

        # First response triggers tool call with invalid JSON
        mock_openai_client.add_chat_response(
            messages=[{"role": "user", "content": "Square 4"}],
            response="Using tool",
            tool_calls=[tool_call_with_invalid_json_arguments],
        )

        state = await env.rollout(
            input=make_input(
                prompt=[{"role": "user", "content": "Square 4"}], answer="", task=""
            ),
            client=mock_openai_client,
            model="test-model",
        )

        # Should have error set
        assert state.get("error") is not None
        assert isinstance(state["error"], vf.ToolParseError)
        assert isinstance(state["error"], vf.ToolError)

        # Should have partial trajectory (one step with the tool call attempt)
        assert len(state["trajectory"]) == 1

        # Should render completion conditions (e.g. is_completed, timing, stop_condition)
        assert state["is_completed"] is True
        assert state["stop_condition"] == "has_error"
        assert state["timing"] is not None
        assert state["completion"] is not None

    @pytest.mark.asyncio
    async def test_tool_env_tool_call_error(
        self, mock_openai_client, sample_chat_dataset, make_input
    ):
        """Test that ToolEnv stops rollout when tool raises an exception."""

        class ErrorToolEnv(vf.ToolEnv):
            def __init__(self, **kwargs):
                super().__init__(tools=[faulty_tool], stop_errors=[Exception], **kwargs)

        env = ErrorToolEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
        )

        tool_call = _build_tool_call("faulty_tool", {})

        mock_openai_client.add_chat_response(
            messages=[{"role": "user", "content": "Invoke"}],
            response="Using tool",
            tool_calls=[tool_call],
        )

        state = await env.rollout(
            input=make_input(
                prompt=[{"role": "user", "content": "Invoke"}], answer="", task=""
            ),
            client=mock_openai_client,
            model="test-model",
        )

        # Should have error set
        assert state.get("error") is not None
        assert isinstance(state["error"], vf.ToolCallError)
        assert isinstance(state["error"], vf.ToolError)

        # Should have partial trajectory (one step with the tool call attempt)
        assert len(state["trajectory"]) == 1

        # Should render completion conditions (e.g. is_completed, timing, stop_condition)
        assert state["is_completed"] is True
        assert state["stop_condition"] == "has_error"
        assert state["timing"] is not None
        assert state["completion"] is not None

    def test_add_tool_no_duplicate(self, mock_openai_client, sample_chat_dataset):
        """Test that add_tool doesn't add duplicate entries to tools list."""
        env = vf.ToolEnv(
            tools=[square_tool],
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
        )

        initial_tool_count = len(env.tools)
        assert initial_tool_count == 1

        env.add_tool(offset_tool)

        assert len(env.tools) == 2
        assert env.tools.count(square_tool) == 1
        assert env.tools.count(offset_tool) == 1

    def test_remove_tool_no_error(self, mock_openai_client, sample_chat_dataset):
        """Test that remove_tool removes a tool exactly once."""
        env = vf.ToolEnv(
            tools=[square_tool, offset_tool],
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
        )

        assert len(env.tools) == 2

        env.remove_tool(square_tool)

        assert len(env.tools) == 1
        assert square_tool not in env.tools
        assert offset_tool in env.tools

    def test_add_tool_updates_tool_monitor_rubric(
        self, mock_openai_client, sample_chat_dataset
    ):
        """Test that add_tool properly updates tool_monitor_rubric metrics."""
        env = vf.ToolEnv(
            tools=[square_tool],
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
        )

        assert "square_tool" in env.tool_monitor_rubric.tool_names
        assert "offset_tool" not in env.tool_monitor_rubric.tool_names

        env.add_tool(offset_tool)

        assert "offset_tool" in env.tool_monitor_rubric.tool_names
        assert len(env.tool_monitor_rubric.tool_names) == 2

    @pytest.mark.asyncio
    async def test_call_tool_returns_valid_text_content_parts(
        self, mock_openai_client, sample_chat_dataset
    ):
        """Test that call_tool preserves valid text content parts."""

        def text_parts_tool() -> list:
            return [{"type": "text", "text": "Hello world"}]

        env = vf.ToolEnv(
            tools=[text_parts_tool],
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
        )

        result = await env.call_tool("text_parts_tool", {}, "call_0")
        assert result["content"] == [{"type": "text", "text": "Hello world"}]

    @pytest.mark.asyncio
    async def test_call_tool_returns_valid_image_url_content_parts(
        self, mock_openai_client, sample_chat_dataset
    ):
        """Test that call_tool preserves valid image_url content parts."""

        def image_tool() -> list:
            return [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ]

        env = vf.ToolEnv(
            tools=[image_tool],
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
        )

        result = await env.call_tool("image_tool", {}, "call_0")
        assert result["content"] == [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        ]

    @pytest.mark.asyncio
    async def test_call_tool_returns_mixed_content_parts(
        self, mock_openai_client, sample_chat_dataset
    ):
        """Test that call_tool preserves mixed valid content parts."""

        def mixed_tool() -> list:
            return [
                {"type": "text", "text": "Here's the screenshot"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ]

        env = vf.ToolEnv(
            tools=[mixed_tool],
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
        )

        result = await env.call_tool("mixed_tool", {}, "call_0")
        assert result["content"] == [
            {"type": "text", "text": "Here's the screenshot"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]

    @pytest.mark.asyncio
    async def test_call_tool_casts_invalid_list_to_str(
        self, mock_openai_client, sample_chat_dataset
    ):
        """Test that call_tool casts invalid lists (not content parts) to str."""

        def list_tool() -> list:
            return [1, 2, 3]

        env = vf.ToolEnv(
            tools=[list_tool],
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
        )

        result = await env.call_tool("list_tool", {}, "call_0")
        assert result["content"] == "[1, 2, 3]"

    @pytest.mark.asyncio
    async def test_call_tool_casts_list_missing_type_to_str(
        self, mock_openai_client, sample_chat_dataset
    ):
        """Test that call_tool casts list with missing type keys to str."""

        def bad_list_tool() -> list:
            return [{"text": "no type key"}]

        env = vf.ToolEnv(
            tools=[bad_list_tool],
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
        )

        result = await env.call_tool("bad_list_tool", {}, "call_0")
        assert result["content"] == "[{'text': 'no type key'}]"

    @pytest.mark.asyncio
    async def test_call_tool_casts_list_with_invalid_type_to_str(
        self, mock_openai_client, sample_chat_dataset
    ):
        """Test that call_tool casts list with invalid type values to str."""

        def invalid_type_tool() -> list:
            return [{"type": "audio", "data": "base64data"}]

        env = vf.ToolEnv(
            tools=[invalid_type_tool],
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
        )

        result = await env.call_tool("invalid_type_tool", {}, "call_0")
        assert result["content"] == "[{'type': 'audio', 'data': 'base64data'}]"
