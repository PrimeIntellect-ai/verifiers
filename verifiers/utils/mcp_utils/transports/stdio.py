from __future__ import annotations

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool

from verifiers.utils.mcp_utils.models import MCPServerConfig
from verifiers.utils.mcp_utils.transports.base import (
    MCPTransport,
    normalize_mcp_tool_result,
)


class StdioTransport(MCPTransport):
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: ClientSession | None = None
        self.tools: dict[str, MCPTool] = {}
        self._connection_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._error: Exception | None = None

    async def connect(self) -> dict[str, MCPTool]:
        self._connection_task = asyncio.create_task(self._maintain_connection())
        await self._ready.wait()

        if self._error is not None:
            raise self._error

        return self.tools

    async def _maintain_connection(self) -> None:
        try:
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
                env=self.config.env,
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()
                    tools_response = await session.list_tools()
                    self.tools = {
                        tool.name: tool for tool in tools_response.tools
                    }
                    self._ready.set()

                    while True:
                        await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            self.session = None
            self.tools = {}

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if self.session is None:
            raise RuntimeError(f"Server '{self.config.name}' not connected")
        result = await self.session.call_tool(tool_name, arguments=arguments)
        return normalize_mcp_tool_result(result)

    async def disconnect(self) -> None:
        if self._connection_task is None:
            return
        self._connection_task.cancel()
        try:
            await self._connection_task
        except asyncio.CancelledError:
            pass
        finally:
            self._connection_task = None
            self.session = None
            self.tools = {}
            self._ready.clear()
            self._error = None

    async def is_connected(self) -> bool:
        return self.session is not None and (
            self._connection_task is not None and not self._connection_task.done()
        )
