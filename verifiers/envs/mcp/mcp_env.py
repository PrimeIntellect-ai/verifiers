import atexit
import signal
from typing import Callable, Dict, List, Literal, Optional

from .transports.base import MCPTransport
from .transports.sandbox import SandboxTransport
from .transports.stdio import StdioTransport
from .transports.streaming_http import StreamingHTTPTransport
from .mcp_utils.models import MCPServerConfig
from .mcp_utils.mcp_tool_wrapper import MCPToolWrapper
from .mcp_utils.mcp_env_utils import validate_config, get_server_url, create_transport

import verifiers as vf
from verifiers import Messages, State


try:
    from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest
except ImportError:
    AsyncSandboxClient = None
    CreateSandboxRequest = None

class MCPEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        mcp_servers: List[MCPServerConfig | dict],
        # transport configuration
        transport_type: Literal["stdio", "http", "sandbox"] = "stdio",
        connection_scope: Literal["rollout", "session"] = "rollout",
        # http specific
        http_urls: Optional[Dict[str, str]] = None,
        http_timeout: float = 30.0,
        http_max_retries: int = 3,
        # sandbox specific
        sandbox_image: str = "python:3.11-slim",
        sandbox_start_command: str = "tail -f /dev/null",
        sandbox_cpu_cores: int = 1,
        sandbox_memory_gb: int = 2,
        sandbox_disk_size_gb: int = 5,
        sandbox_timeout_minutes: int = 60,
        sandbox_environment_vars: Optional[Dict[str, str]] = None,
        sandbox_port_to_expose: Optional[int] = None,
        # standard options
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {str(e)}",
        **kwargs
    ):
        self.mcp_servers = [
            MCPServerConfig(**s) if isinstance(s, dict) else s
            for s in mcp_servers
        ]

        self.transport_type = transport_type
        self.connection_scope = connection_scope
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
        self.sandbox_client: Optional[AsyncSandboxClient] = None
        self.active_sandboxes: set[str] = set()

        validate_config(transport_type, self.mcp_servers, self.connection_scope)

        self.session_transports: Dict[str, MCPTransport] = {}

        super().__init__(
            tools=[],
            max_turns=max_turns,
            error_formatter=error_formatter,
            **kwargs
        )

        if transport_type == "sandbox":
            if AsyncSandboxClient is None:
                raise ImportError(
                    "prime-sandboxes required for sandbox transport "
                )
            self.sandbox_client = AsyncSandboxClient()
            atexit.register(self.cleanup_sandboxes)
            signal.signal(
                signal.SIGINT,
                lambda sig, frame: (
                    self.cleanup_sandboxes(),
                    signal.default_int_handler(sig, frame),
                ),
            )
            signal.signal(
                signal.SIGTERM,
                lambda _, __: (self.cleanup_sandboxes(), exit(143))
            )


    async def register_tools_from_transport(
        self,
        server_name: str,
        transport: MCPTransport
    ):
        """Register tools from an MCP transport, bypassing convert_func_to_oai_tool validation."""
        for tool in transport.tools.values():
            wrapper = MCPToolWrapper(server_name, tool, transport)

            # Manually register without going through add_tool()
            # This avoids convert_func_to_oai_tool() which enforces strict schema validation
            self.tools.append(wrapper)

            # Use wrapper's to_oai_tool() which has schema cleaning
            oai_tool = wrapper.to_oai_tool()

            if self.oai_tools is None:
                self.oai_tools = []
            self.oai_tools.append(oai_tool)

            # Register in tool_map
            tool_name = wrapper.__name__
            self.tool_map[tool_name] = wrapper

            # Also track skipped args (empty for MCP tools)
            if not hasattr(self, 'skipped_args'):
                self.skipped_args = {}
            self.skipped_args[tool_name] = []

    async def setup_session_connections(self):
        for server_config in self.mcp_servers:
            transport = await create_transport(self, server_config)

            if self.transport_type == "sandbox":
                await transport.create_sandbox()
                self.active_sandboxes.add(transport.sandbox_id)
                await transport.run_setup_commands()
                await transport.start_mcp_server()
                await transport.expose_port()

            await transport.connect()

            self.session_transports[server_config.name] = transport

            await self.register_tools_from_transport(server_config.name, transport)

    async def setup_state(self, state: State, **kwargs) -> State:
        state = await super().setup_state(state, **kwargs)

        if self.connection_scope == "session":
            if not self.session_transports:
                await self.setup_session_connections()

        elif self.connection_scope == "rollout":
            rollout_transports = {}

            for server_config in self.mcp_servers:
                transport = await create_transport(self, server_config)

                if self.transport_type == "sandbox":
                    await transport.create_sandbox()
                    self.active_sandboxes.add(transport.sandbox_id)
                    await transport.run_setup_commands()
                    await transport.start_mcp_server()
                    await transport.expose_port()

                await transport.connect()
                print(f"[MCP_ENV] Connected, registering tools...", flush=True)

                rollout_transports[server_config.name] = transport

                await self.register_tools_from_transport(server_config.name, transport)
                print(f"[MCP_ENV] Tools registered", flush=True)

            state["mcp_transports"] = rollout_transports

        if self.oai_tools:
            state["info"]["oai_tools"] = self.oai_tools

        print(f"[MCP_ENV] setup_state complete", flush=True)
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict:
        """
        Override this method to inject/hide tool arguments 
        """
        return tool_args

    async def is_completed(
        self,
        messages: Messages,
        state: State,
        **kwargs
    ) -> bool:
        print(f"[ROLLOUT] checking if completed")
        completed = await super().is_completed(messages, state, **kwargs)

        if completed and self.connection_scope == "rollout":
            print(f"[ROLLOUT] is complete and connection scope rollout, cleaning transports")
            await self._cleanup_rollout_transports(state)

        return completed

    async def _cleanup_rollout_transports(self, state: State):
        rollout_transports = state.get("mcp_transports", {})
        print(f"[ROLLOUT] cleaning up rollout transports: {rollout_transports}", flush=True)

        for transport in rollout_transports.values():
            print(f"[ROLLOUT] disconnecting transport: {transport}", flush=True)
            await transport.disconnect()
            print(f"[ROLLOUT] disconnected transport: {transport}", flush=True)

            if self.transport_type == "sandbox":
                sandbox_transport = transport
                if sandbox_transport.sandbox_id:
                    self.active_sandboxes.discard(sandbox_transport.sandbox_id)

        state.pop("mcp_transports", None)

    async def cleanup(self):
        for transport in self.session_transports.values():
            await transport.disconnect()

            if self.transport_type == "sandbox":
                sandbox_transport = transport
                if sandbox_transport.sandbox_id:
                    self.active_sandboxes.discard(sandbox_transport.sandbox_id)

        self.session_transports.clear()

    def cleanup_sandboxes(self):
        if not self.active_sandboxes:
            return

        self.logger.info(f"Cleaning up {len(self.active_sandboxes)} sandboxes")

        from prime_sandboxes import SandboxClient, APIClient
        client = SandboxClient(APIClient())

        for sandbox_id in list(self.active_sandboxes):
            try:
                client.delete(sandbox_id)
                self.active_sandboxes.discard(sandbox_id)
            except Exception as e:
                self.logger.error(f"Failed to cleanup sandbox {sandbox_id}: {e}")
