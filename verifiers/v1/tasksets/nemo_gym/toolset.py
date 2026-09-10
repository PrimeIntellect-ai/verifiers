"""Expose NeMo Gym resource tools through Verifiers MCP."""

from typing import Any, cast

import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import Field

from verifiers.v1.harnesses.utils.mcp import mcp_client
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
    tool_names: list[str] = Field(default_factory=list)

    async def post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        """POST to Gym while carrying this rollout's cookie session forward."""
        async with httpx.AsyncClient(
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.request_timeout,
        ) as client:
            return await client.post(f"{self.resources_url}/{path}", json=body)


class NeMoGymToolset(Toolset[SharedToolsetConfig, NeMoGymState]):
    """Bridge rollout-specific Gym tools into the standard V1 MCP boundary."""

    TOOL_PREFIX = None

    def register(self, mcp: MCPServer) -> None:
        # MCPServer's protocol handlers call these public methods, so replacing them keeps the
        # dynamic, rollout-specific catalog without registering a fake static tool schema.
        server = cast(Any, mcp)
        server.list_tools = self._with_state(self.list_tools)
        server.call_tool = self._with_state(self.call_tool)

    async def list_tools(self) -> list[Tool]:
        if self.state.mcp_url is not None:
            async with mcp_client(
                {
                    "url": self.state.mcp_url,
                    "headers": self.state.mcp_headers,
                    "timeout": self.state.request_timeout,
                    "connect_timeout": self.state.request_timeout,
                }
            ) as client:
                tools = (await client.list_tools()).tools
        else:
            tools = [
                Tool(
                    name=spec["name"],
                    description=spec.get("description") or None,
                    input_schema=spec.get("parameters") or {},
                )
                for spec in self.state.direct_tools.values()
            ]
        self.state.tool_names = [tool.name for tool in tools]
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context | None = None,
    ) -> CallToolResult:
        if self.state.mcp_url is not None:
            async with mcp_client(
                {
                    "url": self.state.mcp_url,
                    "headers": self.state.mcp_headers,
                    "timeout": self.state.request_timeout,
                    "connect_timeout": self.state.request_timeout,
                }
            ) as client:
                return await client.call_tool(name, arguments)

        if name not in self.state.direct_tools:
            raise ValueError(f"unknown NeMo Gym tool: {name}")

        response = await self.state.post(name, arguments)
        return CallToolResult(
            content=[TextContent(type="text", text=response.text)],
            is_error=not response.is_success,
        )


if __name__ == "__main__":
    NeMoGymToolset.run()
