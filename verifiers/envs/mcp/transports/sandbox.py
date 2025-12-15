import asyncio
from typing import Dict, Optional

from mcp.types import Tool

from .streaming_http import StreamingHTTPTransport
from ..mcp_utils.models import MCPServerConfig


class SandboxTransport(StreamingHTTPTransport):
    def __init__(
        self,
        config: MCPServerConfig,
        sandbox_client,
        sandbox_request,
        port_to_expose: Optional[int] = None,
        **kwargs
    ):
        self.sandbox_client = sandbox_client
        self.sandbox_request = sandbox_request
        self.sandbox_id: Optional[str] = None
        self.port_to_expose: Optional[int] = port_to_expose

        self.port_exposed: bool = False
        self.exposure_id: Optional[str] = None

        super().__init__(config, url="", **kwargs)

    async def create_sandbox(self) -> str:
        """Create the sandbox and wait for it to be ready."""
        sandbox = await self.sandbox_client.create(self.sandbox_request)
        self.sandbox_id = sandbox.id
        await self.sandbox_client.wait_for_creation(self.sandbox_id)
        print(f"[SANDBOX] Sandbox created: {self.sandbox_id}", flush=True)
        return self.sandbox_id

    async def run_command(self, command: str) -> str:
        """Run a command in the sandbox."""
        if not self.sandbox_id:
            raise RuntimeError("Sandbox not created yet")
        result = await self.sandbox_client.execute_command(self.sandbox_id, command)
        return result

    async def run_setup_commands(self) -> None:
        """Run setup commands before starting the MCP server."""
        if not self.config.setup_commands:
            return
        print(f"[SANDBOX] Running setup commands: {self.config.setup_commands}", flush=True)
        for cmd in self.config.setup_commands:
            await self.run_command(cmd)

    async def start_mcp_server(self) -> None:
        """Start the MCP server process in the sandbox."""
        print(f"[SANDBOX] Starting MCP server...", flush=True)
        # Build env vars string
        env_str = ""
        if self.config.env:
            env_str = " ".join(f"{k}={v}" for k, v in self.config.env.items()) + " "

        cmd = f"{self.config.command} {' '.join(self.config.args or [])}"
        bg_cmd = f"{env_str}nohup {cmd} > /tmp/mcp.log 2>&1 &"
        await self.run_command(bg_cmd)
        await asyncio.sleep(3)

    async def expose_port(self) -> str:
        """Expose a port on the sandbox and return the URL."""
        print(f"[SANDBOX] Exposing port {self.port_to_expose}...", flush=True)
        exposed = await self.sandbox_client.expose(self.sandbox_id, self.port_to_expose)
        self.port_exposed = True
        self.exposure_id = exposed.exposure_id
        self.url = f"{exposed.url}/mcp"
        print(f"[SANDBOX] Exposed URL: {self.url}")

        # wait for DNS propagation
        print("[SANDBOX] Waiting 10s for DNS...")
        await asyncio.sleep(10)
        print("[SANDBOX] DNS wait done")
        return self.url

    async def connect(self) -> Dict[str, Tool]:
        """Connect to the MCP server (assumes sandbox is already set up)."""
        print(f"[SANDBOX] Connecting to {self.url}...")
        result = await super().connect()
        print(f"[SANDBOX] Connected! Got {len(result)} tools")
        return result

    async def disconnect(self) -> None:
        """Disconnect from MCP server and delete the sandbox."""
        try:
            await super().disconnect()
        finally:
            if self.sandbox_id:
                await self.sandbox_client.delete(self.sandbox_id)
                self.sandbox_id = None
