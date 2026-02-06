from typing import Callable, Dict, List, Literal, Optional

from verifiers.utils.mcp_utils.transports.base import MCPTransport
from verifiers.utils.mcp_utils.models import MCPServerConfig, MCPTransportConfig
from verifiers.utils.mcp_utils.mcp_tool_wrapper import MCPToolWrapper
from verifiers.utils.mcp_utils.mcp_env_utils import validate_config, create_transport

import verifiers as vf
from verifiers import Messages, State


class MCPEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        mcp_servers: List[MCPServerConfig | dict],
        transport_type: Literal["stdio", "http", "sandbox"] = "stdio",
        http_urls: Optional[Dict[str, str]] = None,
        http_timeout: float = 30.0,
        http_max_retries: int = 3,
        sandbox_image: str = "python:3.11-slim",
        sandbox_start_command: str = "tail -f /dev/null",
        sandbox_cpu_cores: int = 1,
        sandbox_memory_gb: int = 2,
        sandbox_disk_size_gb: int = 5,
        sandbox_timeout_minutes: int = 60,
        sandbox_environment_vars: Optional[Dict[str, str]] = None,
        sandbox_port_to_expose: Optional[int] = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {str(e)}",
        **kwargs
    ):
        self.mcp_servers = [
            MCPServerConfig(**s) if isinstance(s, dict) else s
            for s in mcp_servers
        ]

        self.transport_type = transport_type
        self.http_urls = http_urls or {}
        self.http_timeout = http_timeout
        self.http_max_retries = http_max_retries

        self.sandbox_image = sandbox_image
        self.sandbox_start_command = sandbox_start_command
        self.sandbox_cpu_cores = sandbox_cpu_cores
        self.sandbox_memory_gb = sandbox_memory_gb
        self.sandbox_disk_size_gb = sandbox_disk_size_gb
        self.sandbox_timeout_minutes = sandbox_timeout_minutes
        self.sandbox_environment_vars = sandbox_environment_vars or {}
        self.sandbox_port_to_expose = sandbox_port_to_expose

        validate_config(
            transport_type,
            self.mcp_servers,
            http_urls=self.http_urls,
        )

        self.session_transports: Dict[str, MCPTransport] = {}

        super().__init__(
            tools=[],
            max_turns=max_turns,
            error_formatter=error_formatter,
            **kwargs
        )

    async def register_tools_from_transport(
        self,
        server_name: str,
        transport: MCPTransport,
    ) -> None:
        for tool in transport.tools.values():
            wrapper = MCPToolWrapper(server_name, tool, transport)
            self.tools.append(wrapper)
            oai_tool = wrapper.to_oai_tool()
            if self.oai_tools is None:
                self.oai_tools = []
            self.oai_tools.append(oai_tool)
            tool_name = wrapper.__name__
            self.tool_map[tool_name] = wrapper
            if not hasattr(self, "skipped_args"):
                self.skipped_args = {}
            self.skipped_args[tool_name] = []

    async def setup_session_connections(self) -> None:
        for server_config in self.mcp_servers:
            transport_config = MCPTransportConfig(
                server_config=server_config,
                transport_type=self.transport_type,
                http_urls=self.http_urls,
                http_timeout=self.http_timeout,
                http_max_retries=self.http_max_retries,
                sandbox_image=self.sandbox_image,
                sandbox_start_command=self.sandbox_start_command,
                sandbox_cpu_cores=self.sandbox_cpu_cores,
                sandbox_memory_gb=self.sandbox_memory_gb,
                sandbox_disk_size_gb=self.sandbox_disk_size_gb,
                sandbox_timeout_minutes=self.sandbox_timeout_minutes,
                sandbox_environment_vars=self.sandbox_environment_vars,
                sandbox_port_to_expose=self.sandbox_port_to_expose,
            )
            transport = await create_transport(transport_config)
            if self.transport_type == "sandbox":
                await transport.create_sandbox()
                await transport.run_setup_commands()
                await transport.start_mcp_server()
                await transport.expose_port()

            await transport.connect()
            self.session_transports[server_config.name] = transport
            await self.register_tools_from_transport(server_config.name, transport)

    async def setup_state(self, state: State, **kwargs) -> State:
        state = await super().setup_state(state, **kwargs)

        if not self.session_transports:
            await self.setup_session_connections()

        if self.oai_tools:
            state["info"]["oai_tools"] = self.oai_tools

        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict:
        return tool_args

    async def cleanup(self) -> None:
        for transport in self.session_transports.values():
            await transport.disconnect()
        self.session_transports.clear()
