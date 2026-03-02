from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp.types import Tool as MCPTool

from verifiers.types import Tool


@dataclass(slots=True, frozen=True)
class RegisteredMCPTool:
    public_name: str
    server_name: str
    remote_name: str
    description: str
    parameters: dict[str, object]

    def to_tool_def(self) -> Tool:
        return Tool(
            name=self.public_name,
            description=self.description,
            parameters=self.parameters,
        )


class MCPToolProxy:
    """Callable proxy stored in tool_map for routed MCP tool execution."""

    def __init__(self, registration: RegisteredMCPTool, env: Any):
        self.registration = registration
        self.env = env
        self.__name__ = registration.public_name
        self.__doc__ = registration.description or ""

    async def __call__(self, state, **kwargs):
        return await self.env.call_mcp_tool(
            public_tool_name=self.registration.public_name,
            tool_args=kwargs,
            state=state,
        )


def build_registered_tool(
    *, public_name: str, server_name: str, tool: MCPTool
) -> RegisteredMCPTool:
    parameters = tool.inputSchema or {"type": "object", "properties": {}}
    return RegisteredMCPTool(
        public_name=public_name,
        server_name=server_name,
        remote_name=tool.name,
        description=tool.description or "",
        parameters=parameters,
    )
