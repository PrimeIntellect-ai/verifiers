import atexit
import signal
from typing import Any, Callab,e Dict, List, Literal, Optional

from datasets import Dataset

from transports.base import MCPTransport
from transports.stdio import StdioTransport
from transports.streaming_http import StreamingHTTPTransport
from transports.sandbox import SandboxTransport
from models import MCPServerConfig
from mcp_tool_wrapper import MCPToolWrapper

import verifiers as vf
from verifiers import Message, State


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
        self.sandbox_environment_vars = sandbox_environment_vars
        self.sandbox_client: Optional[AsyncSandboxClient] = None

        self._validate_config()

        self.session_transports: Dict[str, MCPTransport] = {}
        self.active_transports: set[str] = set()

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
            self.register(self.cleanup_sandboxes)
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


