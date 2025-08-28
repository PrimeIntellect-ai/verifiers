import json
from abc import abstractmethod
from typing import Callable

from verifiers.envs.tool_env import ToolEnv
from verifiers.types import ChatCompletionMessageToolCall, Message, Messages, State
from verifiers.utils.tool_utils import convert_func_to_oai_tool


class StatefulToolEnv(ToolEnv):
    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"{str(e)}",
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            max_turns=max_turns,
            error_formatter=error_formatter,
            **kwargs,
        )
        self.tools = tools or []
        self.max_turns = max_turns
        self.error_formatter = error_formatter
        self.oai_tools = [convert_func_to_oai_tool(tool) for tool in self.tools]
        self.tool_map = {tool.__name__: tool for tool in self.tools}

    @abstractmethod
    def update_tool_args(
        self, tool_args: dict, messages: Messages, state: State, **kwargs
    ) -> dict:
        """Update tool arguments and/or state (in-place) based on messages and state."""
        pass

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> Message:
        """Call a tool based on JSON command."""
        try:
            tool_func = self.tool_map[tool_name]
            result = str(tool_func(**tool_args))
            return {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call_id,
            }
        except Exception as e:
            return {
                "role": "tool",
                "content": self.error_formatter(e),
                "tool_call_id": tool_call_id,
            }

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        assert isinstance(messages, list)
        assert "tool_calls" in messages[-1]
        tool_messages = []
        for tool_call in messages[-1]["tool_calls"]:
            assert isinstance(tool_call, ChatCompletionMessageToolCall)
            tool_name: str = tool_call.function.name
            tool_args: dict = json.loads(tool_call.function.arguments)
            tool_call_id: str = tool_call.id or ""
            tool_args = self.update_tool_args(tool_args, messages, state, **kwargs)
            tool_message: Message = await self.call_tool(
                tool_name, tool_args, tool_call_id
            )
            tool_messages.append(tool_message)
        return tool_messages, state
