from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, cast

from verifiers.errors import ToolError
from verifiers.types import Tool

from ..toolset import MCPTool


class MCPToolHandle:
    def __init__(self, session: object, tool_def: Tool):
        self.session = session
        self.name = tool_def.name
        self.tool_def = tool_def

    async def __call__(self, **kwargs: object) -> object:
        result = await cast(Any, self.session).call_tool(self.name, dict(kwargs))
        return mcp_result_value(result)


async def connect_mcp_tool(
    spec: MCPTool, exit_stack: AsyncExitStack
) -> list[MCPToolHandle]:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server = StdioServerParameters(
        command=spec.command,
        args=list(spec.args),
        env=dict(spec.env) if spec.env is not None else None,
        cwd=spec.cwd,
    )
    read_stream, write_stream = await exit_stack.enter_async_context(
        stdio_client(server)
    )
    session = await exit_stack.enter_async_context(
        ClientSession(read_stream, write_stream)
    )
    await session.initialize()
    tools_result = await session.list_tools()
    return [MCPToolHandle(session, mcp_tool_def(tool)) for tool in tools_result.tools]


def mcp_tool_def(tool: object) -> Tool:
    raw = cast(Any, tool)
    schema = getattr(raw, "inputSchema", None) or getattr(raw, "input_schema", None)
    if schema is None and hasattr(raw, "model_dump"):
        dumped = raw.model_dump()
        schema = dumped.get("inputSchema") or dumped.get("input_schema")
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    return Tool(
        name=str(raw.name),
        description=str(getattr(raw, "description", "") or ""),
        parameters=cast(dict[str, object], schema),
        strict=None,
    )


def mcp_result_value(result: object) -> object:
    raw = cast(Any, result)
    if bool(getattr(raw, "isError", False)):
        raise ToolError(str(mcp_content_value(getattr(raw, "content", []))))
    return mcp_content_value(getattr(raw, "content", []))


def mcp_content_value(content: object) -> object:
    if not isinstance(content, list):
        return serializable_content(content)
    values = [serializable_content(item) for item in content]
    if len(values) == 1:
        return values[0]
    return values


def serializable_content(item: object) -> object:
    item_type = getattr(item, "type", None)
    text = getattr(item, "text", None)
    if item_type == "text" and isinstance(text, str):
        return text
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    return item
