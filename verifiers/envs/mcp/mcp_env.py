import atexit
import signal
from typing import Callable, Dict, List, Literal, Optional

from .transports.base import MCPTransport
from .transports.sandbox import SandboxTransport
from .transports.stdio import StdioTransport
from .transports.streaming_http import StreamingHTTPTransport
from .mcp_utils.models import MCPServerConfig
from .mcp_utils.mcp_tool_wrapper import MCPToolWrapper

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

        self._validate_config()

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

    def _get_server_url(self, config: MCPServerConfig) -> str:
        """Get URL for a server, checking config.url first, then http_urls dict."""
        if config.url:
            return config.url
        if config.name in self.http_urls:
            return self.http_urls[config.name]
        raise ValueError(
            f"No URL found for server '{config.name}'. "
            f"Provide either 'url' in server config or add to 'http_urls' dict."
        )

    def _validate_config(self):
        """Validate configuration based on transport type."""
        if self.transport_type == "stdio":
            # stdio requires command
            missing_commands = [
                s.name for s in self.mcp_servers
                if not s.command
            ]
            if missing_commands:
                raise ValueError(
                    f"'command' is required for stdio transport. "
                    f"Missing for servers: {missing_commands}"
                )

        elif self.transport_type == "http":
            # http requires url (either in config or http_urls dict)
            missing_urls = [
                s.name for s in self.mcp_servers
                if not s.url and s.name not in self.http_urls
            ]
            if missing_urls:
                raise ValueError(
                    f"URL required for HTTP transport. "
                    f"Missing for servers: {missing_urls}. "
                    f"Provide 'url' in server config or add to 'http_urls' dict."
                )

        elif self.transport_type == "sandbox":
            # sandbox requires command for starting MCP server in sandbox
            missing_commands = [
                s.name for s in self.mcp_servers
                if not s.command
            ]
            if missing_commands:
                raise ValueError(
                    f"'command' is required for sandbox transport. "
                    f"Missing for servers: {missing_commands}"
                )
            
            if self.connection_scope == "session":
                self.logger.warning(
                    "Using session scope with sandbox transport. "
                    "This will create one sandbox for all rollouts."
                )

    async def create_transport(
        self,
        config: MCPServerConfig,
    ) -> MCPTransport:
        if self.transport_type == "stdio":
            if not config.command:
                raise ValueError(f"'command' required for stdio transport: {config.name}")
            return StdioTransport(config)

        elif self.transport_type == "http":
            url = self._get_server_url(config)
            return StreamingHTTPTransport(
                config,
                url=url,
                timeout=self.http_timeout,
                max_retries=self.http_max_retries
            )

        elif self.transport_type == "sandbox":
            if not self.sandbox_client:
                raise RuntimeError("Sandbox client not initialized")
            if not config.command:
                raise ValueError(f"'command' required for sandbox transport: {config.name}")

            env_vars = {**self.sandbox_environment_vars, **(config.env or {})}

            sandbox_request = CreateSandboxRequest(
                name=f"mcp-{config.name}",
                docker_image=self.sandbox_image,
                start_command=self.sandbox_start_command,
                cpu_cores=self.sandbox_cpu_cores,
                memory_gb=self.sandbox_memory_gb,
                disk_size_gb=self.sandbox_disk_size_gb,
                timeout_minutes=self.sandbox_timeout_minutes,
                environment_vars=env_vars,
            )

            return SandboxTransport(
                config,
                sandbox_client=self.sandbox_client,
                sandbox_request=sandbox_request,
                port_to_expose=self.sandbox_port_to_expose,
                timeout=self.http_timeout,
                max_retries=self.http_max_retries
            )

        else:
            raise ValueError(f"Unknown transport type: {self.transport_type}")

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
        for config in self.mcp_servers:
            transport = await self.create_transport(config)
            await transport.connect()

            self.session_transports[config.name] = transport

            if self.transport_type == "sandbox":
                sandbox_transport = transport
                if sandbox_transport.sandbox_id:
                    self.active_sandboxes.add(sandbox_transport.sandbox_id)

            await self.register_tools_from_transport(config.name, transport)

    async def setup_state(self, state: State, **kwargs) -> State:
        state = await super().setup_state(state, **kwargs)

        if self.oai_tools:
            state["info"]["oai_tools"] = self.oai_tools


        if self.connection_scope == "session":
            if not self.session_transports:
                await self.setup_session_connections()

        elif self.connection_scope == "rollout":
            rollout_transports = {}

            for config in self.mcp_servers:
                transport = await self.create_transport(config)
                await transport.connect()

                rollout_transports[config.name] = transport

                if self.transport_type == "sandbox":
                    sandbox_transport = transport
                    if sandbox_transport.sandbox_id:
                        self.active_sandboxes.add(sandbox_transport.sandbox_id)

                await self.register_tools_from_transport(config.name, transport)

            state["mcp_transports"] = rollout_transports

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
        completed = await super().is_completed(messages, state, **kwargs)

        if completed and self.connection_scope == "rollout":
            await self._cleanup_rollout_transports(state)

        return completed

    async def _cleanup_rollout_transports(self, state: State):
        rollout_transports = state.get("mcp_transports", {})

        for transport in rollout_transports.values():
            await transport.disconnect()

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
