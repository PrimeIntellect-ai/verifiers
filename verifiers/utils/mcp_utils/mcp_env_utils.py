from __future__ import annotations

from verifiers.utils.mcp_utils.models import MCPServerConfig, MCPTransportConfig
from verifiers.utils.mcp_utils.transports.base import MCPTransport
from verifiers.utils.mcp_utils.transports.sandbox import SandboxTransport
from verifiers.utils.mcp_utils.transports.stdio import StdioTransport
from verifiers.utils.mcp_utils.transports.streaming_http import StreamingHTTPTransport


def create_transport(transport_config: MCPTransportConfig) -> MCPTransport:
    if transport_config.transport_type == "stdio":
        if not transport_config.server_config.command:
            raise ValueError(
                f"'command' required for stdio transport: {transport_config.server_config.name}"
            )
        return StdioTransport(transport_config.server_config)

    if transport_config.transport_type == "http":
        url = get_server_url(
            transport_config.server_config, transport_config.http_urls or {}
        )
        return StreamingHTTPTransport(
            transport_config.server_config,
            url=url,
            timeout=transport_config.http_timeout,
            max_retries=transport_config.http_max_retries,
        )

    if transport_config.transport_type == "sandbox":
        return SandboxTransport(
            transport_config.server_config,
            sandbox_image=transport_config.sandbox_image,
            sandbox_start_command=transport_config.sandbox_start_command,
            sandbox_environment_vars=transport_config.sandbox_environment_vars or {},
            sandbox_cpu_cores=transport_config.sandbox_cpu_cores,
            sandbox_memory_gb=transport_config.sandbox_memory_gb,
            sandbox_disk_size_gb=transport_config.sandbox_disk_size_gb,
            sandbox_timeout_minutes=transport_config.sandbox_timeout_minutes,
            port_to_expose=transport_config.sandbox_port_to_expose,
            timeout=transport_config.http_timeout,
            max_retries=transport_config.http_max_retries,
        )

    raise ValueError(f"Unknown transport type: {transport_config.transport_type}")


def get_server_url(server_config: MCPServerConfig, http_urls: dict[str, str]) -> str:
    if server_config.url:
        return server_config.url
    if server_config.name in http_urls:
        return http_urls[server_config.name]
    raise ValueError(
        f"No URL found for server '{server_config.name}'. "
        "Provide either 'url' in the server config or an entry in 'http_urls'."
    )


def validate_config(
    transport_type: str,
    servers: list[MCPServerConfig],
    *,
    http_urls: dict[str, str] | None = None,
) -> None:
    server_names = [server.name for server in servers]
    if len(server_names) != len(set(server_names)):
        raise ValueError("Each MCP server must have a unique 'name'")

    if transport_type == "stdio":
        missing = [server.name for server in servers if not server.command]
        if missing:
            raise ValueError(
                f"'command' required for stdio transport. Missing: {missing}"
            )
        return

    if transport_type == "http":
        urls = http_urls or {}
        missing = [
            server.name
            for server in servers
            if not server.url and server.name not in urls
        ]
        if missing:
            raise ValueError(
                "URL required for http transport "
                "(per-server 'url' or entry in 'http_urls'). "
                f"Missing: {missing}"
            )
        return

    if transport_type == "sandbox":
        missing = [server.name for server in servers if not server.command]
        if missing:
            raise ValueError(
                f"'command' required for sandbox transport. Missing: {missing}"
            )
        return

    raise ValueError(f"Unknown transport type: {transport_type}")
