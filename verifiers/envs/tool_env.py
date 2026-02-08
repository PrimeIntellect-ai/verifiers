import json
from typing import Callable, cast

import verifiers as vf
from verifiers.types import Messages, ToolMessage
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.tool_utils import convert_func_to_tool, is_valid_tool_content_parts


class ToolMonitorRubric(vf.Rubric):
    def __init__(self, tools: list[Callable] | None = None, **kwargs):
        super().__init__(**kwargs)

        self.tools = tools or []
        self.tool_names = [tool.__name__ for tool in self.tools]  # type: ignore[union-attr]

        # add tool metrics
        self.add_metric(self.total_tool_calls)
        for tool_name in self.tool_names:
            self.add_metric(self.get_tool_call_count_func(tool_name))

    def add_tool_metric(self, tool: Callable):
        tool_name = tool.__name__  # type: ignore[union-attr]
        if tool_name not in self.tool_names:
            self.tool_names.append(tool_name)
            self.add_metric(self.get_tool_call_count_func(tool_name))

    def remove_tool_metric(self, tool: Callable):
        tool_name = tool.__name__  # type: ignore[union-attr]
        if tool_name in self.tool_names:
            self.tool_names.remove(tool_name)
            metric_name = f"{tool_name}_calls"
            for i, func in enumerate(self.funcs):
                if func.__name__ == metric_name:
                    self.funcs.pop(i)
                    self.weights.pop(i)
                    break

    async def total_tool_calls(self, completion: Messages) -> float:
        """Count the total number of tool calls."""
        total = 0
        assert isinstance(completion, list)
        for msg in completion:
            if msg.role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
                total += len(msg.tool_calls)
        return float(total)

    def get_tool_call_count_func(self, tool_name: str) -> Callable:
        """Create a metric that counts calls to a specific tool."""

        async def tool_call_count_func(completion: Messages) -> int:
            """Count calls to {tool_name} tool."""
            count = 0
            assert isinstance(completion, list)
            for msg in completion:
                if msg.role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.name == tool_name:
                            count += 1

            return count

        tool_call_count_func.__name__ = f"{tool_name}_calls"
        return tool_call_count_func


class ToolEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"{e}",
        stop_errors: list[type[Exception]] | None = None,
        **kwargs,
    ):
        self.tools = tools or []
        self.max_turns = max_turns
        self.error_formatter = error_formatter
        self.stop_errors: list[type[Exception]] = stop_errors or []
        self.tool_defs = [convert_func_to_tool(tool) for tool in self.tools]
        self.tool_map = {
            getattr(tool, "__name__", tool.__class__.__name__): tool
            for tool in self.tools
        }
        super().__init__(tool_defs=self.tool_defs, max_turns=max_turns, **kwargs)

        self.tool_monitor_rubric = ToolMonitorRubric(tools=self.tools)
        self.add_rubric(self.tool_monitor_rubric)

    def _should_stop_for_error(self, err: Exception) -> bool:
        """Check if error is in stop_errors."""
        return any(isinstance(err, err_type) for err_type in self.stop_errors)

    def add_tool(self, tool: Callable):
        self.tools.append(tool)
        if self.tool_defs is None:
            self.tool_defs = []
        self.tool_defs.append(convert_func_to_tool(tool))
        self.tool_map[getattr(tool, "__name__", tool.__class__.__name__)] = tool
        self.tool_monitor_rubric.add_tool_metric(tool)

    def remove_tool(self, tool: Callable):
        self.tools.remove(tool)
        if self.tool_defs is None:
            self.tool_defs = []
        self.tool_defs.remove(convert_func_to_tool(tool))
        tool_name = getattr(tool, "__name__", tool.__class__.__name__)
        self.tool_map.pop(tool_name)
        self.tool_monitor_rubric.remove_tool_metric(tool)

    @vf.stop
    async def no_tools_called(self, state: vf.State) -> bool:
        if len(state["trajectory"]) == 0:
            return False
        last_message = state["trajectory"][-1]["completion"][-1]
        is_assistant_message = last_message.role == "assistant"
        no_tool_calls = (
            not hasattr(last_message, "tool_calls") or not last_message.tool_calls
        )
        return is_assistant_message and no_tool_calls

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> ToolMessage:
        """Call a tool based on JSON command."""
        tool_func = self.tool_map[tool_name]
        result = await maybe_await(tool_func, **tool_args)
        content = (
            result
            if is_valid_tool_content_parts(result)
            else str(result)
        )
        return ToolMessage(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
        )

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        last_msg = cast(vf.AssistantMessage, messages[-1])
        assert last_msg.tool_calls is not None
        tool_messages = []
        for tool_call in last_msg.tool_calls:
            tool_call_id: str = tool_call.id
            try:
                tool_name: str = tool_call.name
                tool_args: dict = json.loads(tool_call.arguments)
            except Exception as e:
                if self._should_stop_for_error(e):
                    raise vf.ToolParseError from e
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content=self.error_formatter(e),
                        tool_call_id=tool_call_id,
                    )
                )
                continue  # skip tool call below

            try:
                tool_message = await self.call_tool(
                    tool_name, tool_args, tool_call_id
                )
                tool_messages.append(tool_message)
            except Exception as e:
                if self._should_stop_for_error(e):
                    raise vf.ToolCallError from e
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content=self.error_formatter(e),
                        tool_call_id=tool_call_id,
                    )
                )

        return tool_messages
