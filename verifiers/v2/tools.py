"""Tools: python callables exposed to the model, dispatched on tool calls.

A `Toolset` wraps plain callables, derives their JSON schemas via
`agents.function_schema` (same as v1's `convert_func_to_tool_def`), and dispatches
tool calls concurrently. Tool errors become tool messages, not crashes.
"""

import asyncio
import inspect
import json
from typing import Awaitable, Callable

from agents.function_schema import function_schema

from verifiers.v2.types import Tool, ToolCall, ToolMessage

ToolFn = Callable[..., object | Awaitable[object]]


def tool_from_fn(fn: ToolFn) -> Tool:
    schema = function_schema(fn)
    return Tool(
        name=fn.__name__,
        description=schema.description or "",
        parameters=schema.params_json_schema,
        strict=schema.strict_json_schema,
    )


class Toolset:
    def __init__(self, fns: list[ToolFn]) -> None:
        self.fns: dict[str, ToolFn] = {fn.__name__: fn for fn in fns}
        self.tools: list[Tool] = [tool_from_fn(fn) for fn in fns]

    async def dispatch(self, calls: list[ToolCall]) -> list[ToolMessage]:
        async def run(call: ToolCall) -> ToolMessage:
            fn = self.fns.get(call.name)
            if fn is None:
                return ToolMessage(
                    tool_call_id=call.id, content=f"Error: unknown tool {call.name!r}"
                )
            try:
                result = fn(**json.loads(call.arguments or "{}"))
                if inspect.isawaitable(result):
                    result = await result
                content = result if isinstance(result, str) else json.dumps(result)
            except Exception as e:
                content = f"Error: {e}"
            return ToolMessage(tool_call_id=call.id, content=content)

        return await asyncio.gather(*(run(call) for call in calls))
