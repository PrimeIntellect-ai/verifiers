"""Tests for the StatefulToolEnv class."""

import json
from json import JSONDecodeError

import pytest
from openai.types.chat import ChatCompletionUserMessageParam

import verifiers as vf
from tests.conftest import faulty_tool, secret_tool


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


class TestStatefulToolEnv:
    @pytest.mark.asyncio
    async def test_stateful_tool_env_updates_args(
        self, mock_stateful_tool_env, mock_openai_client, make_input
    ):
        tool_call = _build_tool_call("offset_tool", {"x": 5})
        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
        }
        user_message = ChatCompletionUserMessageParam(content="Offset 5", role="user")

        mock_openai_client.add_chat_response(
            messages=[user_message],
            response="Using tool",
            tool_calls=[tool_call],
        )
        mock_openai_client.add_chat_response(
            messages=[
                user_message,
                assistant_message,
                {
                    "role": "tool",
                    "content": "8",
                    "tool_call_id": "call_0",
                },
            ],
            response="Done",
        )

        state = await mock_stateful_tool_env.rollout(
            input=make_input(prompt=[user_message], task="", answer=""),
            client=mock_openai_client,
            model="test-model",
        )
        completion = state["completion"]

        tool_messages = [m for m in completion if m.get("role") == "tool"]
        assert tool_messages and tool_messages[0]["content"] == "8"
        assert state["update_calls"] == 1
        assert state["last_tool_args"]["offset"] == 3

    def test_stateful_tool_env_add_tool_skips_args(self, mock_stateful_tool_env):
        mock_stateful_tool_env.add_tool(secret_tool, args_to_skip=["secret"])

        schema = next(
            tool
            for tool in mock_stateful_tool_env.oai_tools
            if tool["function"]["name"] == "secret_tool"
        )

        assert "secret" not in schema["function"]["parameters"]["properties"]
        assert mock_stateful_tool_env.skipped_args["secret_tool"] == ["secret"]
        assert "secret_tool" in mock_stateful_tool_env.tool_map

    def test_add_tool_skips_dict_type_args(self, mock_stateful_tool_env):
        def tool_with_dict(command: str, state: dict | None = None) -> str:
            return command

        mock_stateful_tool_env.add_tool(tool_with_dict, args_to_skip=["state"])

        schema = next(
            t
            for t in mock_stateful_tool_env.oai_tools
            if t["function"]["name"] == "tool_with_dict"
        )
        assert "state" not in schema["function"]["parameters"]["properties"]

    def test_add_tool_does_not_mutate_original_signature(self, mock_stateful_tool_env):
        """Verify that add_tool with args_to_skip doesn't mutate the original function."""
        import inspect

        def my_tool(command: str, hidden: int, visible: bool = True) -> str:
            """A tool with multiple parameters."""
            return command

        original_params = list(inspect.signature(my_tool).parameters.keys())
        original_annotations = dict(my_tool.__annotations__)

        mock_stateful_tool_env.add_tool(my_tool, args_to_skip=["hidden"])

        # Original function signature should be unchanged
        assert list(inspect.signature(my_tool).parameters.keys()) == original_params
        assert my_tool.__annotations__ == original_annotations
        assert "hidden" in inspect.signature(my_tool).parameters

        # But schema should have hidden removed
        schema = next(
            t
            for t in mock_stateful_tool_env.oai_tools
            if t["function"]["name"] == "my_tool"
        )
        assert "hidden" not in schema["function"]["parameters"]["properties"]
        assert "command" in schema["function"]["parameters"]["properties"]

    def test_add_tool_does_not_mutate_bound_method_signature(
        self, mock_stateful_tool_env
    ):
        """Verify that add_tool with args_to_skip doesn't mutate bound method signatures."""
        import inspect

        class ToolProvider:
            def my_tool(self, command: str, hidden: int, visible: bool = True) -> str:
                """A tool with multiple parameters."""
                return command

        bound_method = ToolProvider().my_tool
        original_params = list(inspect.signature(bound_method).parameters.keys())

        mock_stateful_tool_env.add_tool(bound_method, args_to_skip=["hidden"])

        # Original bound method signature should be unchanged
        assert (
            list(inspect.signature(bound_method).parameters.keys()) == original_params
        )
        assert "hidden" in inspect.signature(bound_method).parameters

        # But schema should have hidden removed
        schema = next(
            t
            for t in mock_stateful_tool_env.oai_tools
            if t["function"]["name"] == "my_tool"
        )
        assert "hidden" not in schema["function"]["parameters"]["properties"]
        assert "command" in schema["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_tool_env_tool_invalid_json_arguments(
        self, mock_openai_client, sample_chat_dataset, make_input
    ):
        """Test that StatefulToolEnv stops rollout when tool call is not JSON-parsable."""

        class ParseErrorStatefulToolEnv(vf.StatefulToolEnv):
            def __init__(self, **kwargs):
                super().__init__(tools=[], stop_errors=[JSONDecodeError], **kwargs)

            def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
                return tool_args

        env = ParseErrorStatefulToolEnv(
            client=mock_openai_client,
            model="test-model",
            dataset=sample_chat_dataset,
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

        class ErrorStatefulToolEnv(vf.StatefulToolEnv):
            def __init__(self, **kwargs):
                super().__init__(tools=[faulty_tool], stop_errors=[Exception], **kwargs)

            def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
                return tool_args

        env = ErrorStatefulToolEnv(
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
