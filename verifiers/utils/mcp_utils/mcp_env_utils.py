from typing import Optional, Union
from verifiers.utils.mcp_utils.models import MCPServerConfig, MCPTransportConfig
from verifiers.utils.mcp_utils.transports.stdio import StdioTransport
from verifiers.utils.mcp_utils.transports.streaming_http import StreamingHTTPTransport
from verifiers.utils.mcp_utils.transports.sandbox import SandboxTransport

async def create_transport(
    transport_config: MCPTransportConfig, 
) -> Union[StdioTransport, StreamingHTTPTransport, SandboxTransport]:
    if transport_config.transport_type == "stdio":
        if not transport_config.server_config.command:
            raise ValueError(f"'command' required for stdio transport: {transport_config.server_config.name}")
        return StdioTransport(transport_config.server_config)

    elif transport_config.transport_type == "http":
        url = get_server_url(transport_config.server_config, transport_config.http_urls or {})
        return StreamingHTTPTransport(
            transport_config.server_config,
            url=url,
            timeout=transport_config.http_timeout,
            max_retries=transport_config.http_max_retries
        )

    elif transport_config.transport_type == "sandbox":
        from prime_sandboxes import CreateSandboxRequest
        if not transport_config.server_config.command:
            raise ValueError(f"'command' required for sandbox transport: {transport_config.server_config.name}")

        env_vars = {**(transport_config.sandbox_environment_vars or {}), **(transport_config.server_config.env or {})}

        sandbox_request = CreateSandboxRequest(
            name=f"mcp-{transport_config.server_config.name}",
            docker_image=transport_config.sandbox_image,
            start_command=transport_config.sandbox_start_command,
            cpu_cores=transport_config.sandbox_cpu_cores,
            memory_gb=transport_config.sandbox_memory_gb,
            disk_size_gb=transport_config.sandbox_disk_size_gb,
            timeout_minutes=transport_config.sandbox_timeout_minutes,
            environment_vars=env_vars,
        )

        return SandboxTransport(
            transport_config.server_config,
            sandbox_request=sandbox_request,
            port_to_expose=transport_config.sandbox_port_to_expose,
            timeout=transport_config.http_timeout,
            max_retries=transport_config.http_max_retries
        )

    else:
        raise ValueError(f"Unknown transport type: {transport_config.transport_type}")


def get_server_url(server_config: MCPServerConfig, http_urls: dict) -> str:
        """Get URL for a server, checking config.url first, then http_urls dict."""
        if server_config.url:
            return server_config.url
        if server_config.name in http_urls:
            return http_urls[server_config.name]
        raise ValueError(
            f"No URL found for server '{server_config.name}'. "
            f"Provide either 'url' in server config or add to 'http_urls' dict."
        )

def validate_config(
    transport_type: str,
    servers: list[MCPServerConfig],
    connection_scope: str,
    http_urls: Optional[dict[str, str]] = None,
) -> None:
    if transport_type == "stdio":
        missing = [s.name for s in servers if not s.command]
        if missing:
            raise ValueError(f"'command' required for stdio. Missing: {missing}")

    elif transport_type == "http":
        urls = http_urls or {}
        missing = [
            s.name
            for s in servers
            if not s.url and s.name not in urls
        ]
        if missing:
            raise ValueError(
                f"URL required for http (per-server 'url' or entry in 'http_urls'). Missing: {missing}"
            )

    elif transport_type == "sandbox":
        missing = [s.name for s in servers if not s.command]
        if missing:
            raise ValueError(f"'command' required for sandbox. Missing: {missing}")
