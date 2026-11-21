"""Adapt Harbor's task-declared MCP servers to the harness's HTTP transport."""

import asyncio
import os
import secrets
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, cast

from mcp import Client
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, Tool
from pydantic_config import BaseConfig

from verifiers.v1.harnesses.utils.mcp import MCPConnection
from verifiers.v1.harnesses.utils.remote_runtime import prepare_remote_bridge
from verifiers.v1.mcp.auth import MCPURL
from verifiers.v1.mcp.launch import install_in_runtime, serve_in_runtime
from verifiers.v1.mcp.toolset import WorkspaceToolset
from verifiers.v1.runtimes import Runtime


class HarborMCPConfig(BaseConfig):
    server: dict[str, Any]
    """One declaration already validated by Harbor's task parser."""


class HarborMCPAdapter(WorkspaceToolset[HarborMCPConfig]):
    @property
    def server_name(self) -> str:
        return self.config.server["name"]

    @asynccontextmanager
    async def serve_workspace(
        self, harness: Runtime, endpoint: str, secret: str
    ) -> AsyncIterator[str]:
        connection = harness._execution_bridge
        if connection is None:
            raise RuntimeError("Harbor MCP requires a workspace execution connection")
        workspace = connection.runtime
        spec = self.config.server
        if spec["transport"] == "stdio":
            worker = [spec["command"], *spec.get("args", [])]
        else:
            # Task-local HTTP and SSE URLs must resolve inside the workspace.
            python = (
                sys.executable
                if workspace.type == "subprocess"
                else await install_in_runtime(type(self), workspace, label="Harbor MCP")
            )
            worker = [python, "-m", __name__, "--stdio", self.config.model_dump_json()]
        name = f"harbor-{uuid.uuid4().hex}"
        connection.register(name, worker, {})
        command = await prepare_remote_bridge(harness, endpoint, secret, name)
        # Only this trusted relay command is launched in the harness. The task receives
        # its workspace environment and never sees the evaluator or MCP credential.
        frontend = type(self)(
            HarborMCPConfig(
                server={
                    "name": name,
                    "transport": "stdio",
                    "command": command[0],
                    "args": command[1:],
                }
            )
        )
        token = secrets.token_urlsafe(32)
        async with AsyncExitStack() as stack:
            port = await serve_in_runtime(
                frontend, harness, stack=stack, exposed=False, mcp_secret=token
            )
            yield MCPURL(f"http://127.0.0.1:{port}/mcp", token, harness_local=True)

    async def setup(self) -> None:
        # Discovery first happens after task egress is locked. The shared connection
        # then owns the same upstream session across downstream HTTP reconnects.
        self.upstream = MCPConnection(self.config.server, connect=_harbor_client)
        self._exit_stack.push_async_callback(self.upstream.aclose)

    def register(self, mcp: MCPServer) -> None:
        # MCPServer calls these public methods; forward the real catalog and schemas.
        server = cast(Any, mcp)
        server.list_tools = self.list_tools
        server.call_tool = self.call_tool

    async def list_tools(self) -> list[Tool]:
        client = await self.upstream.client()
        tools: list[Tool] = []
        cursor = None
        seen_cursors: set[str] = set()
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return tools
            if cursor in seen_cursors:
                raise ValueError("MCP tools pagination returned a repeated cursor")
            seen_cursors.add(cursor)

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None
    ) -> CallToolResult:
        if name not in {tool.name for tool in await self.list_tools()}:
            raise ToolError(
                f"MCP tool {name!r} is not listed by server {self.server_name!r}"
            )
        client = await self.upstream.client()
        return await client.call_tool(name, arguments)


def _harbor_client(spec: dict) -> Client:
    match spec["transport"]:
        case "stdio":
            # The trusted relay needs the harness's Docker callback proxy.
            proxy = {
                k: v
                for k, v in os.environ.items()
                if k.lower()
                in {
                    "http_proxy",
                    "https_proxy",
                    "all_proxy",
                    "no_proxy",
                    "ssl_cert_file",
                }
            }
            transport = stdio_client(
                StdioServerParameters(
                    command=spec["command"],
                    args=spec.get("args", []),
                    env=proxy,
                )
            )
        case "sse":
            transport = sse_client(spec["url"])
        case "streamable-http":
            transport = streamable_http_client(spec["url"])
        case other:
            raise ValueError(f"Unsupported Harbor MCP transport: {other!r}")
    return Client(transport)


async def _stdio(config: str) -> None:
    adapter = HarborMCPAdapter(HarborMCPConfig.model_validate_json(config))
    async with adapter._exit_stack:
        await adapter.setup()
        mcp = MCPServer(adapter.server_name)
        adapter.register(mcp)
        await mcp.run_stdio_async()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--stdio"]:
        asyncio.run(_stdio(sys.argv[2]))
    else:
        HarborMCPAdapter.run()
