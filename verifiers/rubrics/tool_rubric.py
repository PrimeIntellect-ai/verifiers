from typing import Callable

from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages
from verifiers.utils.tool_utils import convert_func_to_oai_tool


class ToolRubric(Rubric):
    """Simple rubric that counts tool calls in completion messages."""

    def __init__(self, tools: list[Callable] | None = None):
        self.tools = tools or []
        self.oai_tools = [convert_func_to_oai_tool(tool) for tool in self.tools]
        self.tool_names = [tool.__name__ for tool in self.tools]

        # Build initial reward functions and weights
        reward_funcs = [self.total_tool_calls]
        reward_weights = [0.0]

        for tool_name in self.tool_names:
            reward_funcs.append(self.get_tool_call_count_func(tool_name))
            reward_weights.append(0.0)

        # Pass them to parent class
        super().__init__(funcs=reward_funcs, weights=reward_weights)

    def total_tool_calls(self, completion: Messages, **kwargs) -> float:
        """Count the total number of tool calls across all assistant messages."""
        total = 0
        for msg in completion:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                tool_calls = msg.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    total += len(tool_calls)
        return float(total)

    def get_tool_call_count_func(self, tool_name: str) -> Callable:
        """Create a reward function that counts calls to a specific tool."""

        def tool_call_count_func(completion: Messages, **kwargs) -> float:
            """Count calls to {tool_name} tool."""
            count = 0

            # Find tool calls in assistant messages
            for msg in completion:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    tool_calls = msg.get("tool_calls", [])
                    if not isinstance(tool_calls, list):
                        continue

                    for tool_call in tool_calls:
                        if hasattr(tool_call, "function") and hasattr(
                            tool_call.function, "name"
                        ):
                            if tool_call.function.name == tool_name:
                                count += 1

            return float(count)

        tool_call_count_func.__name__ = f"{tool_name}_calls"
        return tool_call_count_func
