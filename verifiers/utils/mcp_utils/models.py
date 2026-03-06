from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class MCPServerConfig:
    name: str
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    description: str = ""
    url: str | None = None
    setup_commands: list[str] | None = None


@dataclass(slots=True)
class MCPTransportConfig:
    server_config: MCPServerConfig
    transport_type: Literal["stdio", "http", "sandbox"]
    http_urls: dict[str, str] | None = None
    http_timeout: float = 30.0
    http_max_retries: int = 3
    sandbox_image: str = "python:3.11-slim"
    sandbox_start_command: str = "tail -f /dev/null"
    sandbox_environment_vars: dict[str, str] | None = None
    sandbox_cpu_cores: int = 1
    sandbox_memory_gb: int = 2
    sandbox_disk_size_gb: int = 5
    sandbox_timeout_minutes: int = 60
    sandbox_port_to_expose: int = 8000
