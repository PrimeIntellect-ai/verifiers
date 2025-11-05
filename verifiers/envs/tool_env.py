import json
from typing import Any, Callable

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Message, Messages, State
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.tool_utils import convert_func_to_oai_tool


class ToolEnv(MultiTurnEnv):
    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"{str(e)}",
        **kwargs,
    ):
        self.tools = tools or []
        self.max_turns = max_turns
        self.error_formatter = error_formatter
        self.oai_tools = [convert_func_to_oai_tool(tool) for tool in self.tools]
        self.tool_map = {
            getattr(tool, "__name__", tool.__class__.__name__): tool
            for tool in self.tools
        }
        super().__init__(oai_tools=self.oai_tools, max_turns=max_turns, **kwargs)

    def add_tool(self, tool: Callable):
        self.tools.append(tool)
        if self.oai_tools is None:
            self.oai_tools = []
        self.oai_tools.append(convert_func_to_oai_tool(tool))
        self.tool_map[getattr(tool, "__name__", tool.__class__.__name__)] = tool

    def remove_tool(self, tool: Callable):
        self.tools.remove(tool)
        if self.oai_tools is None:
            self.oai_tools = []
        self.oai_tools.remove(convert_func_to_oai_tool(tool))
        tool_name = getattr(tool, "__name__", tool.__class__.__name__)
        self.tool_map.pop(tool_name)

    async def is_completed(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> bool:
        completed = await super().is_completed(messages, state, **kwargs)
        assert isinstance(messages, list)
        is_assistant_message = messages[-1]["role"] == "assistant"
        no_tool_calls = (
            "tool_calls" not in messages[-1] or messages[-1]["tool_calls"] is None
        )
        return completed or (is_assistant_message and no_tool_calls)

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> Message:
        """Call a tool based on JSON command."""
        try:
            tool_func = self.tool_map[tool_name]
            result = await maybe_await(tool_func, **tool_args)
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
            tool_name: str = tool_call.get("function", {}).get("name", "")
            tool_args: dict = json.loads(
                tool_call.get("function", {}).get("arguments", "")
            )
            tool_call_id: str = tool_call.get("id", "")
            tool_message: Message = await self.call_tool(
                tool_name, tool_args, tool_call_id
            )
            tool_messages.append(tool_message)
        return tool_messages, state
