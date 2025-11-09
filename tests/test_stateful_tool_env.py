"""Tests for the StatefulToolEnv class."""

import json

import pytest

from tests.conftest import secret_tool
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


class TestStatefulToolEnv:
    @pytest.mark.asyncio
    async def test_stateful_tool_env_updates_args(
        self, mock_stateful_tool_env, mock_openai_client
    ):
        tool_call = _build_tool_call("offset_tool", {"x": 5})
        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call],
        }
        user_message = {"role": "user", "content": "Offset 5"}

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
            input=RolloutInput(
                prompt=[user_message],
                answer="",
                example_id=0,
            ),
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
