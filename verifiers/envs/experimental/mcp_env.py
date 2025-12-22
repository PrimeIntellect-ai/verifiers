import asyncio
import atexit
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, Tool

import verifiers as vf


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str] | None = None
    env: Dict[str, str] | None = None
    description: str = ""


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


class MCPToolWrapper:
    def __init__(
        self, server_name: str, tool: Tool, server_connection: MCPServerConnection
    ):
        self.server_name = server_name
        self.tool = tool
        self.server_connection = server_connection

        self.__name__ = tool.name
        self.__doc__ = tool.description or ""

        self.__annotations__ = self._build_annotations()

    def _build_annotations(self) -> dict:
        annotations = {}

        if self.tool.inputSchema:
            properties = self.tool.inputSchema.get("properties", {})

            for param_name, param_spec in properties.items():
                param_type = param_spec.get("type", "string")
                if param_type == "string":
                    annotations[param_name] = str
                elif param_type == "integer":
                    annotations[param_name] = int
                elif param_type == "number":
                    annotations[param_name] = float
                elif param_type == "boolean":
                    annotations[param_name] = bool
                elif param_type == "array":
                    annotations[param_name] = list
                elif param_type == "object":
                    annotations[param_name] = dict
                else:
                    annotations[param_name] = Any

        annotations["return"] = str
        return annotations

    async def __call__(self, **kwargs):
        return await self.server_connection.call_tool(self.tool.name, kwargs)

    def to_oai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.__name__,
                "description": self.__doc__ or "",
                "parameters": self.tool.inputSchema
                or {"type": "object", "properties": {}},
            },
        }


class MCPEnv(vf.ToolEnv):
    """Environment for MCP-based tools using the official MCP SDK."""

    def __init__(
        self,
        mcp_servers: List[MCPServerConfig | dict] = [],
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {str(e)}",
        **kwargs,
    ):
        self.mcp_servers: List[MCPServerConfig] = []
        if mcp_servers:
            for server in mcp_servers:
                if isinstance(server, MCPServerConfig):
                    self.mcp_servers.append(server)
                else:
                    self.mcp_servers.append(
                        MCPServerConfig(
                            name=server["name"],
                            command=server["command"],
                            args=server.get("args"),
                            env=server.get("env"),
                            description=server.get("description", ""),
                        )
                    )

        self.server_connections: Dict[str, MCPServerConnection] = {}
        self.mcp_tools: Dict[str, MCPToolWrapper] = {}

        self.error_formatter = error_formatter
        self._setup_complete = False
        self._init_kwargs = kwargs
        self._max_turns = max_turns

        super().__init__(
            tools=[], max_turns=max_turns, error_formatter=error_formatter, **kwargs
        )
        # Start a persistent background event loop and connect synchronously
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._run_loop, args=(self._bg_loop,), daemon=True
        )
        self._bg_thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect_servers(), self._bg_loop)
        fut.result()
        self._setup_complete = True

        # cleanup on exit
        atexit.register(
            lambda: (
                asyncio.run_coroutine_threadsafe(self.cleanup(), self._bg_loop).result(
                    timeout=5
                ),
                self._shutdown_loop(),
            )
        )

    def _run_loop(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _connect_servers(self):
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
                    f"Registered MCP tool: {wrapper.__name__} from server '{server_config.name}'"
                )

        self.tools = wrapper_tools
        self.oai_tools = [tool.to_oai_tool() for tool in wrapper_tools]
        self.tool_map = {tool.__name__: tool for tool in wrapper_tools}

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> vf.Message:
        if tool_name in self.tool_map:
            tool_wrapper = self.tool_map[tool_name]
            try:
                result = await tool_wrapper(**tool_args)
                return cast(
                    vf.Message,
                    {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call_id,
                    },
                )
            except Exception as e:
                return cast(
                    vf.Message,
                    {
                        "role": "tool",
                        "content": self.error_formatter(e),
                        "tool_call_id": tool_call_id,
                    },
                )
        else:
            return cast(
                vf.Message,
                {
                    "role": "tool",
                    "content": f"Error: Tool '{tool_name}' not found",
                    "tool_call_id": tool_call_id,
                },
            )

    async def cleanup(self):
        for connection in self.server_connections.values():
            await connection.disconnect()

        self.server_connections.clear()
        self.mcp_tools.clear()

    def _shutdown_loop(self):
        self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        self._bg_thread.join(timeout=5)
