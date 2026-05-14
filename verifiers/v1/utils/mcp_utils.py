import asyncio
from contextlib import AsyncExitStack
from typing import cast

from verifiers.errors import ToolError
from verifiers.types import Tool

from ..toolset import MCPTool
from ..types import ConfigData


class MCPToolHandle:
    def __init__(self, session: "MCPToolSession", tool_def: Tool):
        self.session = session
        self.name = tool_def.name
        self.tool_def = tool_def

    async def __call__(self, **kwargs: object) -> object:
        result = await self.session.call_tool(self.name, dict(kwargs))
        return mcp_result_value(result)


class MCPToolSession:
    def __init__(self, spec: MCPTool):
        self.spec = spec
        self.handles: list[MCPToolHandle] = []
        self._queue: asyncio.Queue[tuple[str, str, ConfigData, asyncio.Future]] = (
            asyncio.Queue()
        )
        self._ready: asyncio.Future[list[MCPToolHandle]] | None = None
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "MCPToolSession":
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._task = loop.create_task(self._run())
        self.handles = await self._ready
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def call_tool(self, name: str, arguments: ConfigData) -> object:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(("call", name, arguments, future))
        return await future

    async def close(self) -> None:
        task = self._task
        if task is None or task.done():
            return
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(("close", "", {}, future))
        await future
        await task

    async def _run(self) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server = StdioServerParameters(
            command=self.spec.command,
            args=list(self.spec.args),
            env=dict(self.spec.env) if self.spec.env is not None else None,
            cwd=self.spec.cwd,
        )
        ready = self._ready
        if ready is None:
            raise RuntimeError("MCPToolSession started without a ready future.")
        try:
            async with stdio_client(server) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    self.handles = [
                        MCPToolHandle(self, mcp_tool_def(tool))
                        for tool in tools_result.tools
                    ]
                    ready.set_result(self.handles)
                    while True:
                        action, name, arguments, future = await self._queue.get()
                        try:
                            if action == "close":
                                cast(asyncio.Future[None], future).set_result(None)
                                return
                            if action != "call":
                                raise RuntimeError(f"Unknown MCP action: {action}")
                            result = await session.call_tool(name, arguments)
                            cast(asyncio.Future, future).set_result(result)
                        except BaseException as exc:
                            cast(asyncio.Future, future).set_exception(exc)
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise


async def connect_mcp_tool(
    spec: MCPTool, exit_stack: AsyncExitStack
) -> list[MCPToolHandle]:
    session = await exit_stack.enter_async_context(MCPToolSession(spec))
    return session.handles


def mcp_tool_def(tool: object) -> Tool:
    schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None)
    model_dump = getattr(tool, "model_dump", None)
    if schema is None and callable(model_dump):
        dumped = model_dump()
        schema = dumped.get("inputSchema") or dumped.get("input_schema")
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    name = getattr(tool, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError("MCP tools require a name.")
    return Tool(
        name=name,
        description=str(getattr(tool, "description", "") or ""),
        parameters=cast(ConfigData, schema),
        strict=None,
    )


def mcp_result_value(result: object) -> object:
    content = getattr(result, "content", [])
    if bool(getattr(result, "isError", False)):
        raise ToolError(str(mcp_content_value(content)))
    return mcp_content_value(content)


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
