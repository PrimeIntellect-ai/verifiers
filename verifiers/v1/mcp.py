from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from verifiers.errors import ToolError
from verifiers.types import Tool

from .toolset import Toolset
from .utils.json_utils import jsonable

if TYPE_CHECKING:
    from mcp import ClientSession

CONTEXT_PARAMETER = "_vf"


class MCPToolRegistry:
    def __init__(
        self,
        toolsets: list[Toolset],
        *,
        parents: list["MCPToolRegistry"] | None = None,
        expose_tools: bool = True,
    ) -> None:
        self.toolsets = toolsets
        self.parents = parents or []
        self.expose_tools = expose_tools
        self._stack = AsyncExitStack()
        self._dispatch: dict[str, tuple[ClientSession, str, bool]] = {}
        self._tools: list[Tool] = []
        self._context: dict[str, object] = {}

    def set_context(self, context: dict[str, object]) -> None:
        self._context = context
        for parent in self.parents:
            parent.set_context(context)

    async def __aenter__(self) -> "MCPToolRegistry":
        if not self.toolsets:
            return self
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.streamable_http import streamablehttp_client

        for toolset in self.toolsets:
            server = toolset.server
            if server.url is not None:
                read, write, *_ = await self._stack.enter_async_context(
                    streamablehttp_client(
                        server.url,
                        headers=server.headers or None,
                    )
                )
            else:
                command = server.command
                if not command:
                    raise ValueError(f"Toolset {toolset.name!r} has no command.")
                read, write = await self._stack.enter_async_context(
                    stdio_client(
                        StdioServerParameters(
                            command=command[0],
                            args=command[1:],
                            env=server.env or None,
                        )
                    )
                )
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            for raw_tool in (await session.list_tools()).tools:
                tool_name = getattr(raw_tool, "name", "")
                if not isinstance(tool_name, str) or not tool_name:
                    raise TypeError("MCP tools require a non-empty name.")
                exposed_name = f"{toolset.name}_{tool_name}"
                if exposed_name in self._dispatch:
                    raise ValueError(f"MCP tool {exposed_name!r} is defined twice.")
                schema = getattr(raw_tool, "inputSchema", None)
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                visible_schema, wants_context = model_visible_schema(schema)
                if tool_visible(toolset, tool_name):
                    self._tools.append(
                        Tool(
                            name=exposed_name,
                            description=str(getattr(raw_tool, "description", "") or ""),
                            parameters=visible_schema,
                        )
                    )
                self._dispatch[exposed_name] = (session, tool_name, wants_context)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    def tool_defs(self, *, include_hidden: bool = False) -> list[Tool] | None:
        tools: list[Tool] = []
        for parent in self.parents:
            tools.extend(parent.tool_defs(include_hidden=include_hidden) or [])
        if self.expose_tools or include_hidden:
            tools.extend(self._tools)
        return tools or None

    async def call(self, name: str, arguments: dict[str, object]) -> object:
        if name not in self._dispatch:
            for parent in self.parents:
                try:
                    return await parent.call(name, arguments)
                except ToolError:
                    pass
            raise ToolError(f"Unknown MCP tool {name!r}.")
        session, raw_name, wants_context = self._dispatch[name]
        payload = dict(arguments)
        if wants_context:
            payload[CONTEXT_PARAMETER] = self._context
        result = await session.call_tool(raw_name, payload)
        if bool(getattr(result, "isError", False)):
            raise ToolError(str(mcp_content_value(getattr(result, "content", []))))
        return mcp_content_value(getattr(result, "content", []))

    async def call_hidden(self, raw_name: str, arguments: dict[str, object]) -> object:
        for parent in self.parents:
            try:
                return await parent.call_hidden(raw_name, arguments)
            except ToolError:
                pass
        matches = [
            exposed_name
            for exposed_name, (_, candidate, _) in self._dispatch.items()
            if candidate == raw_name
        ]
        if not matches:
            raise ToolError(f"Unknown hidden MCP tool {raw_name!r}.")
        if len(matches) > 1:
            raise ToolError(f"Hidden MCP tool {raw_name!r} is ambiguous.")
        return await self.call(matches[0], arguments)

    def has_hidden(self, raw_name: str) -> bool:
        if any(candidate == raw_name for _, candidate, _ in self._dispatch.values()):
            return True
        return any(parent.has_hidden(raw_name) for parent in self.parents)


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
        return jsonable(model_dump(mode="json", exclude_none=True))
    return jsonable(item)


def tool_visible(toolset: Toolset, tool_name: str) -> bool:
    if toolset.show is not None:
        return tool_name in toolset.show
    if toolset.hide is not None:
        return tool_name not in toolset.hide
    return True


def model_visible_schema(schema: dict[str, object]) -> tuple[dict[str, object], bool]:
    if schema.get("type") != "object":
        return dict(schema), False
    properties = schema.get("properties")
    if not isinstance(properties, dict) or CONTEXT_PARAMETER not in properties:
        return dict(schema), False
    visible = dict(schema)
    visible["properties"] = {
        str(key): value for key, value in properties.items() if key != CONTEXT_PARAMETER
    }
    required = schema.get("required")
    if isinstance(required, list):
        visible["required"] = [item for item in required if item != CONTEXT_PARAMETER]
    return visible, True
