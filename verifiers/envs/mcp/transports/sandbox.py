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
        port_to_expose,
        **kwargs
    ):
        self.sandbox_client = sandbox_client
        self.sandbox_request = sandbox_request
        self.sandbox_id: Optional[str] = None
        self.port_to_expose: Optional[int] = port_to_expose

        self.port_exposed: bool = False
        self.url: str = None
        self.exposure_id: str = None

        super().__init__(config, url="", **kwargs)


    async def connect(self) -> Dict[str, Tool]:
        sandbox = await self.sandbox_client.create(self.sandbox_request)
        self.sandbox_id = sandbox.id

        await self.sandbox_client.wait_for_creation(self.sandbox_id)

        start_cmd = f"{self.config.command} {' '.join(self.config.args or [])}"
        await self.sandbox_client.execute_command(
            self.sandbox_id,
            start_cmd,
            background=True
        )

        if self.port_to_expose:
            await self.expose_sandbox()

        return await super().connect()

    async def expose_sandbox(self) -> None:
        exposed_sandbox = await self.sandbox_client.expose(
            self.sandbox_id,
            self.port_to_expose
        )
        # dns for port exposure takes some time so we wait
        await asyncio.sleep(10)
        self.port_exposed = True
        self.url = exposed_sandbox.url
        self.exposure_id = exposed_sandbox.exposure_id


    async def disconnect(self) -> None:
        await super().disconnect()

        if self.sandbox_id:
            await self.sandbox_client.delete(self.sandbox_id)
            self.sandbox_id = None


