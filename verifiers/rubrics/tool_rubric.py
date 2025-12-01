from typing import Callable, cast

from openai.types.chat import ChatCompletionAssistantMessageParam

from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages
from verifiers.utils.tool_utils import convert_func_to_oai_tool


class ToolRubric(Rubric):
    """Simple rubric that counts tool calls in completion messages."""

    def __init__(self, tools: list[Callable] | None = None):
        self.tools = tools or []
        self.oai_tools = [convert_func_to_oai_tool(tool) for tool in self.tools]
        self.tool_names = [tool.__name__ for tool in self.tools]  # type: ignore[union-attr]

        # Build initial reward functions and weights
        reward_funcs = []
        reward_funcs.append(self.total_tool_calls)
        reward_weights = [0.0]

        for tool_name in self.tool_names:
            reward_funcs.append(self.get_tool_call_count_func(tool_name))
            reward_weights.append(0.0)

        # Pass them to parent class
        super().__init__(funcs=reward_funcs, weights=reward_weights)

    async def total_tool_calls(self, completion: Messages) -> float:
        """Count the total number of tool calls across all assistant messages."""
        total = 0
        assert isinstance(completion, list)
        for msg in completion:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                assistant_msg = cast(ChatCompletionAssistantMessageParam, msg)
                tool_calls = assistant_msg.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    total += len(tool_calls)
        return float(total)

    def get_tool_call_count_func(self, tool_name: str) -> Callable:
        """Create a reward function that counts calls to a specific tool."""

        async def tool_call_count_func(completion: Messages) -> float:
            """Count calls to {tool_name} tool."""
            count = 0
            # Find tool calls in assistant messages
            assert isinstance(completion, list)
            for msg in completion:
                if msg["role"] == "assistant" and "tool_calls" in msg:
                    assistant_msg = cast(ChatCompletionAssistantMessageParam, msg)
                    tool_calls = assistant_msg.get("tool_calls", [])
                    for tool_call in tool_calls:
                        if tool_call.get("function", {}).get("name") == tool_name:
                            count += 1

            return float(count)

        tool_call_count_func.__name__ = f"{tool_name}_calls"
        return tool_call_count_func
