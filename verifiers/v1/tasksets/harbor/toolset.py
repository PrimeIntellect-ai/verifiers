"""Task-declared Harbor MCP tools served through the standard VF toolset lifecycle."""

import os
from typing import Any, cast

from mcp import Client
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, Tool

from verifiers.v1.mcp import Toolset, ToolsetConfig


class HarborMCPConfig(ToolsetConfig):
    server: dict[str, Any]
    """One declaration already validated by Harbor's task parser."""


class HarborMCPToolset(Toolset[HarborMCPConfig]):
    @property
    def server_name(self) -> str:
        return self.config.server["name"]

    async def setup(self) -> None:
        spec = self.config.server
        match spec["transport"]:
            case "stdio":
                # Keep task variables, excluding the wrapper's state and bind controls.
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if key
                    not in {
                        "VF_CONFIG",
                        "VF_STATE_URL",
                        "VF_STATE_SECRET",
                        "MCP_HOST",
                        "MCP_PORT",
                        "MCP_PORT_FILE",
                    }
                }
                transport = stdio_client(
                    StdioServerParameters(
                        command=spec["command"], args=spec.get("args", []), env=env
                    )
                )
            case "sse":
                transport = sse_client(spec["url"])
            case "streamable-http":
                transport = streamable_http_client(spec["url"])
            case other:
                raise ValueError(f"Unsupported Harbor MCP transport: {other!r}")
        # ServerBase enters and closes this stack in the same task. Keep upstream
        # session state across downstream client reconnects for the whole rollout.
        self.client = await self._exit_stack.enter_async_context(Client(transport))

    def register(self, mcp: MCPServer) -> None:
        # MCPServer calls these public methods; forward the real catalog and schemas.
        server = cast(Any, mcp)
        server.list_tools = self.list_tools
        server.call_tool = self.call_tool

    async def list_tools(self) -> list[Tool]:
        tools: list[Tool] = []
        cursor = None
        while True:
            page = await self.client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None
    ) -> CallToolResult:
        return await self.client.call_tool(name, arguments)


if __name__ == "__main__":
    HarborMCPToolset.run()
