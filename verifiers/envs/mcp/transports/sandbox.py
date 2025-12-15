import asyncio
import atexit
import signal
from typing import Dict, Optional
from mcp.types import Tool
from prime_sandboxes import SandboxClient, APIClient
from .streaming_http import StreamingHTTPTransport
from ..mcp_utils.models import MCPServerConfig

_active_sandboxes = set()
_cleanup_registered = False

def _register_cleanup():
    global _cleanup_registered
    if _cleanup_registered:
        return
    
    def cleanup():
        if not _active_sandboxes:
            return
        client = SandboxClient(APIClient())
        for sandbox_id in list(_active_sandboxes):
            try:
                client.delete(sandbox_id)
            except Exception:
                pass
        _active_sandboxes.clear()
    
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), signal.default_int_handler(s, f)))
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), exit(143)))
    _cleanup_registered = True



class SandboxTransport(StreamingHTTPTransport):
    _async_client: Optional["AsyncSandboxClient"] = None

    @classmethod
    def get_client(cls) -> "AsyncSandboxClient":
        if cls._async_client is None:
            from prime_sandboxes import AsyncSandboxClient
            cls._async_client = AsyncSandboxClient()
            _register_cleanup()
        return cls._async_client

    def __init__(
        self,
        config: MCPServerConfig,
        sandbox_request,
        port_to_expose: Optional[int] = None,
        **kwargs
    ):
        self.sandbox_request = sandbox_request
        self.sandbox_id: Optional[str] = None
        self.port_to_expose: Optional[int] = port_to_expose
        super().__init__(config, url="", **kwargs)

    async def create_sandbox(self) -> str:
        client = self.get_client()
        sandbox = await client.create(self.sandbox_request)
        self.sandbox_id = sandbox.id
        await client.wait_for_creation(self.sandbox_id)
        _active_sandboxes.add(self.sandbox_id)
        return self.sandbox_id

    async def run_command(self, command: str) -> str:
        if not self.sandbox_id:
            raise RuntimeError("Sandbox not created yet")
        client = self.get_client()
        result = await client.execute_command(self.sandbox_id, command)
        return result

    async def run_setup_commands(self) -> None:
        if not self.config.setup_commands:
            return
        for cmd in self.config.setup_commands:
            await self.run_command(cmd)

    async def start_mcp_server(self) -> None:
        """Start the MCP server process in the sandbox."""
        cmd = f"{self.config.command} {' '.join(self.config.args or [])}"
        bg_cmd = f"nohup {cmd} > /tmp/mcp.log 2>&1 &"
        await self.run_command(bg_cmd)
        await asyncio.sleep(3)

    async def expose_port(self) -> str:
        client = self.get_client()
        exposed = await client.expose(
            self.sandbox_id, 
            self.port_to_expose
        )
        self.url = f"{exposed.url}/mcp"
        # TODO: remove this when we have a better way to wait for the port to be exposed
        await asyncio.sleep(10)
        return self.url

    async def connect(self) -> Dict[str, Tool]:
        return await super().connect()

    async def disconnect(self) -> None:
        try:
            await super().disconnect()
        finally:
            if self.sandbox_id:
                client = self.get_client()
                await client.delete(self.sandbox_id)
                _active_sandboxes.discard(self.sandbox_id)
                self.sandbox_id = None

    async def delete_all_sandboxes(self) -> None:
        client = self.get_client()
        await client.bulk_delete(list(_active_sandboxes))
        _active_sandboxes.clear()
