"""Multi-turn ToolEnv wrapper for MCP stdio servers."""

from __future__ import annotations

import asyncio
import atexit
import threading
from typing import Callable, Dict, Sequence

from verifiers.envs.tool_env import ToolEnv
from verifiers.types import Message

from .mcp import MCPServerConfig, MCPServerConnection, MCPToolWrapper


class MCPEnv(ToolEnv):
    """Environment that proxies MCP stdio servers into ToolEnv tools."""

    def __init__(
        self,
        mcp_servers: Sequence[MCPServerConfig] | None = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] | None = None,
        **kwargs,
    ):
        self.mcp_servers: list[MCPServerConfig] = [
            server if isinstance(server, MCPServerConfig) else MCPServerConfig(**server)
            for server in (mcp_servers or [])
        ]
        self.server_connections: Dict[str, MCPServerConnection] = {}
        self.mcp_tools: Dict[str, MCPToolWrapper] = {}

        formatter = error_formatter or (lambda exc: f"Error: {exc}")

        super().__init__(
            tools=[],
            max_turns=max_turns,
            error_formatter=formatter,
            **kwargs,
        )

        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._run_loop, args=(self._bg_loop,), daemon=True
        )
        self._bg_thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect_servers(), self._bg_loop)
        fut.result()

        self._closed = False
        atexit.register(self._cleanup_at_exit)

    def _cleanup_at_exit(self) -> None:
        self.close()

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _connect_servers(self) -> None:
        wrapper_tools = []

        for server_config in self.mcp_servers:
            connection = MCPServerConnection(server_config, self.logger)
            tools = await connection.connect()

            self.server_connections[server_config.name] = connection

            for tool in tools.values():
                wrapper = MCPToolWrapper(server_config.name, tool, connection)
                wrapper_tools.append(wrapper)
                self.mcp_tools[wrapper.__name__] = wrapper
                self.logger.info(
                    "Registered MCP tool '%s' from server '%s'",
                    wrapper.__name__,
                    server_config.name,
                )

        self.tools = wrapper_tools
        self.oai_tools = [tool.to_oai_tool() for tool in wrapper_tools]
        self.tool_map = {tool.__name__: tool for tool in wrapper_tools}

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> Message:
        if tool_name in self.tool_map:
            tool_wrapper = self.tool_map[tool_name]
            try:
                result = await tool_wrapper(**tool_args)
                return {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call_id,
                }
            except Exception as exc:  # noqa: BLE001
                return {
                    "role": "tool",
                    "content": self.error_formatter(exc),
                    "tool_call_id": tool_call_id,
                }
        return {
            "role": "tool",
            "content": f"Error: Tool '{tool_name}' not found",
            "tool_call_id": tool_call_id,
        }

    async def cleanup(self) -> None:
        for connection in self.server_connections.values():
            await connection.disconnect()

        self.server_connections.clear()
        self.mcp_tools.clear()

    def _shutdown_loop(self) -> None:
        if self._bg_loop.is_running():
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        self._bg_thread.join(timeout=5)

    def close(self) -> None:
        """Synchronously tear down background connections."""

        if self._closed:
            return
        asyncio.run_coroutine_threadsafe(self.cleanup(), self._bg_loop).result()
        self._shutdown_loop()
        self._closed = True
