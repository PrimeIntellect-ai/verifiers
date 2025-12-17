"""Tests for the ToolEnv class."""

import json
from json import JSONDecodeError

import pytest
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

import verifiers as vf
from tests.conftest import faulty_tool, square_tool
from verifiers.types import RolloutInput


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
    async def test_tool_env_calls_tool(self, mock_tool_env, mock_openai_client):
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
            input=RolloutInput(
                prompt=[user_message],
                answer="",
                task="",
                example_id=0,
            ),
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
        self, mock_tool_env, mock_openai_client
    ):
        mock_openai_client.add_chat_response(
            messages=[{"role": "user", "content": "Hello"}],
            response="Hi",
        )

        state = await mock_tool_env.rollout(
            input=RolloutInput(
                prompt=[{"role": "user", "content": "Hello"}],
                answer="",
                task="",
                example_id=0,
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
        self, mock_openai_client, sample_chat_dataset
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
            input=RolloutInput(
                prompt=[{"role": "user", "content": "Square 4"}],
                answer="",
                task="",
                example_id=0,
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
        self, mock_openai_client, sample_chat_dataset
    ):
        """Test that ToolEnv stops rollout when tool raises an exception."""

        class ErrorToolEnv(vf.ToolEnv):
            def __init__(self, **kwargs):
                super().__init__(
                    tools=[faulty_tool], stop_errors=[vf.ToolCallError], **kwargs
                )

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
            input=RolloutInput(
                prompt=[{"role": "user", "content": "Invoke"}],
                answer="",
                task="",
                example_id=0,
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
