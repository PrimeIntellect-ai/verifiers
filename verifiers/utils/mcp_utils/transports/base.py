from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mcp.types import TextContent, Tool as MCPTool


class MCPTransport(ABC):
    tools: dict[str, MCPTool]

    @abstractmethod
    async def connect(self) -> dict[str, MCPTool]:
        """Connect and return the server's tools."""

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Execute a tool call."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect and clean up any transport resources."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Return whether the transport is currently connected."""


def normalize_mcp_tool_result(result: Any) -> str:
    content = getattr(result, "content", None)
    if not content:
        return "No result returned from tool"

    text_parts: list[str] = []
    for item in content:
        if isinstance(item, TextContent):
            text_parts.append(item.text)
            continue
        if getattr(item, "type", None) == "text" and hasattr(item, "text"):
            text_parts.append(str(item.text))
            continue
        text_parts.append(str(item))

    return "\n".join(text_parts) if text_parts else "No result returned from tool"
