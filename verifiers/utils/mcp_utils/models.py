from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from prime_sandboxes import AsyncSandboxClient


@dataclass
class MCPServerConfig:
    name: str
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None  # For HTTP transport - can specify here instead of http_urls dict
    setup_commands: Optional[List[str]] = None  # Commands to run before starting the server (for sandbox transport)


@dataclass
class MCPTransportConfig:
    server_config: MCPServerConfig
    transport_type: Literal["stdio", "http", "sandbox"]
    # http specific
    http_urls: Optional[Dict[str, str]] = None
    http_timeout: float = 30.0
    http_max_retries: int = 3
    # sandbox specific
    sandbox_client: Optional["AsyncSandboxClient"] = None
    sandbox_image: str = "python:3.11-slim"
    sandbox_start_command: str = "tail -f /dev/null"
    sandbox_environment_vars: Optional[Dict[str, str]] = None
    sandbox_cpu_cores: int = 1
    sandbox_memory_gb: int = 2
    sandbox_disk_size_gb: int = 5
    sandbox_timeout_minutes: int = 60
    sandbox_port_to_expose: Optional[int] = 8000
