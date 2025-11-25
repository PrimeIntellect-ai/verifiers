import asyncio
from typing import Dict, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
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
        """Initialize the Streamable HTTP transport."""
        self.config = config
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, Tool] = {}
        self._connection_task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self._error: Optional[Exception] = None

    async def connect(self) -> Dict[str, Tool]:
        """Connect by creating a background task that manages the HTTP connection."""
        self._connection_task = asyncio.create_task(self._maintain_connection())
        await self._ready.wait()
        
        if self._error:
            raise self._error
        
        return self.tools

    async def _maintain_connection(self):
        """Background task that maintains the Streamable HTTP connection."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Use streamablehttp_client instead of sse_client
                async with streamablehttp_client(self.url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        self.session = session
                        await session.initialize()
                        
                        tools_response = await session.list_tools()
                        self.tools = {tool.name: tool for tool in tools_response.tools}
                        
                        self._ready.set()
                        
                        # Keep alive until cancelled
                        try:
                            while True:
                                await asyncio.sleep(1)
                        except asyncio.CancelledError:
                            pass  # Normal shutdown
                            
            except asyncio.CancelledError:
                # Normal cancellation during shutdown
                break
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                else:
                    # Final attempt failed
                    self._error = ConnectionError(
                        f"Failed to connect to MCP server at {self.url} after {self.max_retries} attempts. "
                        f"Ensure the server is running and accessible. Last error: {str(e)}"
                    )
                    self._ready.set()
                    break
        
        # Cleanup
        self.session = None
        self.tools = {}

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call through the MCP session."""
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
        """Disconnect by cancelling the background connection task."""
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        
        self.session = None
        self.tools = {}
        self._ready.clear()

    async def is_connected(self) -> bool:
        """Check if the transport is currently connected."""
        return self.session is not None and (
            self._connection_task is not None and not self._connection_task.done()
        )
