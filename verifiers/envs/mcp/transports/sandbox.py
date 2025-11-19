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
        **kwargs
    ):
        self.sandbox_client = sandbox_client
        self.sandbox_request = sandbox_request
        self.sandbox_id: Optional[str] = None

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

        # TODO (Christian): update when port exposed is available in sandboxes
        self.url = f"http://{self.sandbox_id}.sandboxes.example.com:3000"

        return await super().connect()


    async def disconnect(self) -> None:
        await super().disconnect()

        if self.sandbox_id:
            await self.sandbox_client.delete(self.sandbox_id)
            self.sandbox_id = None


