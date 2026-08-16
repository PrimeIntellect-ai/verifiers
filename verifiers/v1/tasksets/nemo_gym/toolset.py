"""Expose NeMo Gym resource tools through Verifiers MCP."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import (
    create_mcp_http_client,
    streamable_http_client,
)
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import Field

from verifiers.v1.mcp import SharedToolsetConfig, Toolset
from verifiers.v1.state import State


class NeMoGymState(State):
    """Per-rollout session data shared by task hooks and the shared tool server."""

    resources_url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    request_timeout: float = 60.0
    cookies: dict[str, str] = Field(default_factory=dict)
    mcp_url: str | None = None
    mcp_headers: dict[str, str] = Field(default_factory=dict)
    direct_tools: dict[str, dict[str, Any]] = Field(default_factory=dict)

    async def post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        """POST to Gym while carrying this rollout's cookie session forward."""
        async with httpx.AsyncClient(
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.request_timeout,
        ) as client:
            return await client.post(f"{self.resources_url}/{path}", json=body)

    @asynccontextmanager
    async def mcp_session(self) -> AsyncIterator[ClientSession]:
        """Open an upstream MCP session using this rollout's credentials."""
        assert self.mcp_url is not None
        stack = AsyncExitStack()
        try:
            client = await stack.enter_async_context(
                create_mcp_http_client(
                    headers=self.mcp_headers,
                    timeout=httpx.Timeout(self.request_timeout),
                )
            )
            read, write, *_ = await stack.enter_async_context(
                streamable_http_client(self.mcp_url, http_client=client)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            yield session
        finally:
            with suppress(Exception):
                await stack.aclose()


class NeMoGymToolset(Toolset[SharedToolsetConfig, NeMoGymState]):
    """Bridge rollout-specific Gym tools into the standard V1 MCP boundary."""

    TOOL_PREFIX = "nemo_gym"

    def register(self, mcp: FastMCP) -> None:
        server = mcp._mcp_server
        server.list_tools()(self._with_state(self.list_tools))
        server.call_tool(validate_input=False)(self._with_state(self.call_tool))

    async def list_tools(self) -> list[Tool]:
        if self.state.mcp_url is not None:
            async with self.state.mcp_session() as session:
                tools = (await session.list_tools()).tools
        else:
            tools = [
                Tool(
                    name=spec["name"],
                    description=spec.get("description") or None,
                    inputSchema=spec.get("parameters") or {},
                )
                for spec in self.state.direct_tools.values()
            ]
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        if self.state.mcp_url is not None:
            async with self.state.mcp_session() as session:
                return await session.call_tool(name, arguments)

        if name not in self.state.direct_tools:
            raise ValueError(f"unknown NeMo Gym tool: {name}")

        response = await self.state.post(name, arguments)
        return CallToolResult(
            content=[TextContent(type="text", text=response.text)],
            isError=not response.is_success,
        )


if __name__ == "__main__":
    NeMoGymToolset.run()
