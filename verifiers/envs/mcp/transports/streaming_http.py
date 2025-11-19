from typing import Dict, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import TextContent, Tool

from .base import MCPTransport
from ..mcp_utils.models import MCPServerConfig

class StreamingHTTPTransport(MCPTransport):
    def __init__(
        self, 
        config: MCPServerConfig,
        url: str,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.config = config
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, Tool] = {}
        self._sse_context = None
        self._session_context = None


    async def connect(self) -> Dict[str, Tool]:
        self._sse_context = sse_client(self.url)
        read, write = await self._sse_context.__aenter__()

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

        if self._sse_context:
            await self._sse_context.__aexit__(None, None, None)
            self._sse_context = None

        self.tools = {}

    async def is_connected(self) -> bool:
        return self.session is not None
