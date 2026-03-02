from __future__ import annotations

import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool as MCPTool

from verifiers.utils.mcp_utils.models import MCPServerConfig
from verifiers.utils.mcp_utils.transports.base import (
    MCPTransport,
    normalize_mcp_tool_result,
)


class StreamingHTTPTransport(MCPTransport):
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.config = config
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: ClientSession | None = None
        self.tools: dict[str, MCPTool] = {}
        self._connection_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._error: Exception | None = None

    async def connect(self) -> dict[str, MCPTool]:
        self._connection_task = asyncio.create_task(self._maintain_connection())
        await self._ready.wait()

        if self._error is not None:
            raise self._error

        return self.tools

    async def _maintain_connection(self) -> None:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                async with streamablehttp_client(self.url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        self.session = session
                        await asyncio.wait_for(
                            session.initialize(), timeout=self.timeout
                        )
                        tools_response = await asyncio.wait_for(
                            session.list_tools(), timeout=self.timeout
                        )
                        self.tools = {
                            tool.name: tool for tool in tools_response.tools
                        }
                        self._ready.set()

                        while True:
                            await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(float(attempt + 1))
                    continue
                self._error = ConnectionError(
                    f"Failed to connect to MCP server at {self.url} after "
                    f"{self.max_retries} attempts. Last error: {exc}"
                )
                self._ready.set()
                break
            finally:
                if self._error is not None:
                    self.session = None
                    self.tools = {}

        if self._error is None and last_error is not None and not self._ready.is_set():
            self._error = last_error
            self._ready.set()

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if self.session is None:
            raise RuntimeError(f"Server '{self.config.name}' not connected")
        result = await asyncio.wait_for(
            self.session.call_tool(tool_name, arguments=arguments),
            timeout=self.timeout,
        )
        return normalize_mcp_tool_result(result)

    async def disconnect(self) -> None:
        if self._connection_task is not None:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        self._connection_task = None
        self.session = None
        self.tools = {}
        self._ready.clear()
        self._error = None

    async def is_connected(self) -> bool:
        return self.session is not None and (
            self._connection_task is not None and not self._connection_task.done()
        )
