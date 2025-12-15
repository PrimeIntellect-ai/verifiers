from typing import TYPE_CHECKING
from verifiers.envs.mcp.mcp_utils.models import MCPServerConfig
from verifiers.envs.mcp.transports.base import MCPTransport
from verifiers.envs.mcp.transports.stdio import StdioTransport
from verifiers.envs.mcp.transports.streaming_http import StreamingHTTPTransport
from verifiers.envs.mcp.transports.sandbox import SandboxTransport
from prime_sandboxes import CreateSandboxRequest
import logging

if TYPE_CHECKING:
    from verifiers.envs.mcp.mcp_env import MCPEnv

logger = logging.getLogger(__name__)


async def create_transport(
    env: "MCPEnv",
    server_config: MCPServerConfig,
) -> MCPTransport:
    if env.transport_type == "stdio":
        if not server_config.command:
            raise ValueError(f"'command' required for stdio transport: {server_config.name}")
        return StdioTransport(server_config)

    elif env.transport_type == "http":
        url = get_server_url(server_config, env.http_urls or {})
        return StreamingHTTPTransport(
            server_config,
            url=url,
            timeout=env.http_timeout,
            max_retries=env.http_max_retries
        )

    elif env.transport_type == "sandbox":
        if not env.sandbox_client:
            raise RuntimeError("Sandbox client not initialized")
        if not server_config.command:
            raise ValueError(f"'command' required for sandbox transport: {server_config.name}")

        env_vars = {**(env.sandbox_environment_vars or {}), **(server_config.env or {})}

        sandbox_request = CreateSandboxRequest(
            name=f"mcp-{server_config.name}",
            docker_image=env.sandbox_image,
            start_command=env.sandbox_start_command,
            cpu_cores=env.sandbox_cpu_cores,
            memory_gb=env.sandbox_memory_gb,
            disk_size_gb=env.sandbox_disk_size_gb,
            timeout_minutes=env.sandbox_timeout_minutes,
            environment_vars=env_vars,
        )

        return SandboxTransport(
            server_config,
            sandbox_client=env.sandbox_client,
            sandbox_request=sandbox_request,
            port_to_expose=env.sandbox_port_to_expose,
            timeout=env.http_timeout,
            max_retries=env.http_max_retries
        )

    else:
        raise ValueError(f"Unknown transport type: {env.transport_type}")


def get_server_url(config: MCPServerConfig, http_urls: dict) -> str:
        """Get URL for a server, checking config.url first, then http_urls dict."""
        if config.url:
            return config.url
        if config.name in http_urls:
            return http_urls[config.name]
        raise ValueError(
            f"No URL found for server '{config.name}'. "
            f"Provide either 'url' in server config or add to 'http_urls' dict."
        )

def validate_config(transport_type: str, mcp_servers: list[MCPServerConfig], connection_scope: str) -> None:
        """Validate configuration based on transport type."""
        if transport_type == "stdio":
            # stdio requires command
            missing_commands = [
                s.name for s in mcp_servers
                if not s.command
            ]
            if missing_commands:
                raise ValueError(
                    f"'command' is required for stdio transport. "
                    f"Missing for servers: {missing_commands}"
                )

        elif transport_type == "http":
            # http requires url (either in config or http_urls dict)
            missing_urls = [
                s.name for s in mcp_servers
                if not s.url and s.name not in http_urls
            ]
            if missing_urls:
                raise ValueError(
                    f"URL required for HTTP transport. "
                    f"Missing for servers: {missing_urls}. "
                    f"Provide 'url' in server config or add to 'http_urls' dict."
                )

        elif transport_type == "sandbox":
            # sandbox requires command for starting MCP server in sandbox
            missing_commands = [
                s.name for s in mcp_servers
                if not s.command
            ]
            if missing_commands:
                raise ValueError(
                    f"'command' is required for sandbox transport. "
                    f"Missing for servers: {missing_commands}"
                )
            
            if connection_scope == "session":
                logger.warning(
                    "Using session scope with sandbox transport. "
                    "This will create one sandbox for all rollouts."
                )
