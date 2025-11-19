import asyncio
from typing import Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, Tool

from .base import MCPTransport
from ..mcp_utils.models import MCPServerConfig

class StdioTransport(MCPTransport):
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, Tool] = {}
        self._stdio_context = None
        self._session_context = None


    async def connect(self) -> Dict[str, Tool]:
        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args or [],
            env=self.config.env,
        )
        self._stdio_context = stdio_client(server_params)
        read, write = await self._stdio_context.__aenter__()

        self._session_context = ClientSession(read, write)
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()

        tools_response = await self.session.list_tools()
        self.tools = {tool.name: tool for tool in tools_response.tools}

        return self.tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self.session:
            raise RuntimeError(f"Server '{self.config.name}' not connected")

        result = await self.session.call_tool(tool_name, arguments=arguments)

        if result.content:
            text_parts = []
            for content_item in result.content:
                if isinstance(content_item, TextContent):
                    text_parts.append(content_item.text)
                else:
                    text_parts.append(str(content_item))
            return "\n".join(text_parts)

        return "No result returned from tool"


    async def disconnect(self) -> None:
        if self._session_context and self.session:
            await self._session_context.__aexit__(None, None, None)
            self.session = None

        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)
            self._stdio_context = None

        self.tools = {}

    async def is_connected(self) -> bool:
        return self.session is not None
