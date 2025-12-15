from abc import ABC, abstractmethod
from typing import Dict
from mcp.types import Tool

class MCPTransport(ABC):
    """Base transport for MCP connections."""

    @abstractmethod
    async def connect(self) -> Dict[str, Tool]:
        """Connect and return available tools."""
        pass

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanup connection"""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check connection status"""
        pass


