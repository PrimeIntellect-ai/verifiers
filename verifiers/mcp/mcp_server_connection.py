import asyncio
import logging
from typing import Dict, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, Tool

from .models import MCPServerConfig


class MCPServerConnection:
    def __init__(self, config: MCPServerConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, Tool] = {}

        self._connection_task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self._error: Optional[Exception] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self):
        # Record the loop this connection is bound to
        self.loop = asyncio.get_running_loop()
        self._connection_task = asyncio.create_task(self._get_connection())

        await self._ready.wait()

        if self._error:
            raise self._error

        return self.tools

    async def _get_connection(self):
        try:
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
                env=self.config.env,
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session

                    await session.initialize()

                    tools_response = await session.list_tools()

                    for tool in tools_response.tools:
                        self.tools[tool.name] = tool

                    self._ready.set()

                    while True:
                        await asyncio.sleep(1)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._error = e
            self._ready.set()
        finally:
            self.session = None
            self.tools = {}

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        assert self.session is not None, f"Server '{self.config.name}' not connected"
        assert self.loop is not None, "Connection loop not initialized"
        fut = asyncio.run_coroutine_threadsafe(
            self.session.call_tool(tool_name, arguments=arguments), self.loop
        )
        result = await asyncio.wrap_future(fut)

        if result.content:
            text_parts = []
            for content_item in result.content:
                if hasattr(content_item, "text"):
                    assert isinstance(content_item, TextContent)
                    text_parts.append(content_item.text)
                elif hasattr(content_item, "type") and content_item.type == "text":
                    text_parts.append(getattr(content_item, "text", str(content_item)))
                else:
                    text_parts.append(str(content_item))

            return "\n".join(text_parts)

        return "No result returned from tool"

    async def disconnect(self):
        assert self._connection_task is not None
        self._connection_task.cancel()
        try:
            await self._connection_task
        except asyncio.CancelledError:
            pass
        self.logger.info(f"MCP server '{self.config.name}' terminated")
