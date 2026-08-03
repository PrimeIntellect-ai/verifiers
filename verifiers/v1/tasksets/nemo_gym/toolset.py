"""Expose NeMo Gym resource tools through Verifiers MCP."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import Field

from verifiers.v1.mcp import SharedToolsetConfig, Toolset
from verifiers.v1.state import State

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult
    from mcp.types import Tool as MCPTool


class NeMoGymState(State):
    """Per-rollout session data shared by task hooks and the shared tool server."""

    resources_url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    request_timeout: float = 60.0
    cookies: dict[str, str] = Field(default_factory=dict)
    mcp_url: str | None = None
    mcp_headers: dict[str, str] = Field(default_factory=dict)
    direct_tools: list[dict[str, Any]] = Field(default_factory=list)


async def _post(state: NeMoGymState, path: str, body: dict[str, Any]) -> httpx.Response:
    """POST to Gym while carrying the rollout's cookie session forward."""
    async with httpx.AsyncClient(
        headers=state.headers,
        cookies=state.cookies,
        timeout=state.request_timeout,
    ) as client:
        response = await client.post(f"{state.resources_url}/{path}", json=body)
    state.cookies.update(response.cookies)
    return response


@contextlib.asynccontextmanager
async def _mcp_session(
    url: str, headers: dict[str, str], timeout: float
) -> AsyncIterator[ClientSession]:
    """Open and close an upstream MCP session in the caller's task."""
    from mcp import ClientSession
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )

    stack = contextlib.AsyncExitStack()
    try:
        client = await stack.enter_async_context(
            create_mcp_http_client(headers=headers, timeout=httpx.Timeout(timeout))
        )
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(url, http_client=client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        yield session
    finally:
        with contextlib.suppress(Exception):
            await stack.aclose()


class NeMoGymToolset(Toolset[SharedToolsetConfig, NeMoGymState]):
    """Bridge rollout-specific Gym tools into the standard V1 MCP boundary."""

    TOOL_PREFIX = None

    def _register(self, mcp: FastMCP) -> None:
        server = mcp._mcp_server
        server.list_tools()(self._with_state(self.list_tools))
        server.call_tool(validate_input=False)(self._with_state(self.call_tool))

    async def list_tools(self) -> list[MCPTool]:
        from mcp.types import Tool

        if self.state.mcp_url is not None:
            async with _mcp_session(
                self.state.mcp_url,
                self.state.mcp_headers,
                self.state.request_timeout,
            ) as session:
                return (await session.list_tools()).tools
        return [
            Tool(
                name=spec["name"],
                description=spec.get("description") or None,
                inputSchema=spec.get("parameters") or {},
            )
            for spec in self.state.direct_tools
            if spec.get("type") == "function"
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        from mcp.types import CallToolResult, TextContent

        if self.state.mcp_url is not None:
            async with _mcp_session(
                self.state.mcp_url,
                self.state.mcp_headers,
                self.state.request_timeout,
            ) as session:
                return await session.call_tool(name, arguments)

        if not any(
            spec.get("type") == "function" and spec.get("name") == name
            for spec in self.state.direct_tools
        ):
            raise ValueError(f"unknown NeMo Gym tool: {name}")

        response = await _post(self.state, name, arguments)
        return CallToolResult(
            content=[TextContent(type="text", text=response.text)],
            isError=not response.is_success,
        )


if __name__ == "__main__":
    NeMoGymToolset.run()
