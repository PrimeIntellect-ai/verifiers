from __future__ import annotations

from typing import Any

from mcp.types import Tool as MCPTool

from verifiers.utils.mcp_utils.transports.base import MCPTransport


class SyntheticTransport(MCPTransport):
    """In-memory transport used for testing and deterministic backends."""

    def __init__(
        self,
        *,
        tools: dict[str, MCPTool],
        handlers: dict[str, Any],
        data: dict[str, Any] | None = None,
        name: str = "synthetic",
    ):
        self.name = name
        self.tools = tools
        self.handlers = handlers
        self.data = data if data is not None else {}
        self._connected = False

    async def connect(self) -> dict[str, MCPTool]:
        self._connected = True
        return self.tools

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        if not self._connected:
            raise RuntimeError(f"Transport '{self.name}' not connected")
        if tool_name not in self.handlers:
            raise ValueError(f"No handler registered for tool '{tool_name}'")
        result = self.handlers[tool_name](self.data, arguments)
        if hasattr(result, "__await__"):
            result = await result
        return result

    async def disconnect(self) -> None:
        self._connected = False

    async def is_connected(self) -> bool:
        return self._connected


def create_tool(
    *,
    name: str,
    description: str,
    parameters: dict[str, Any],
    required: list[str] | None = None,
) -> MCPTool:
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": parameters,
    }
    if required:
        input_schema["required"] = required
    return MCPTool(
        name=name,
        description=description,
        inputSchema=input_schema,
    )
