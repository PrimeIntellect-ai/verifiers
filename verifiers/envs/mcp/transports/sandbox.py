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

        if self.port_to_expose:
            await self.expose_sandbox()

        start_cmd = f"nohup {self.config.command} {' '.join(self.config.args or [])} > /tmp/mcp.log 2>&1 &"
        await self.sandbox_client.execute_command(
            self.sandbox_id,
            start_cmd
        )

        # give the MCP server time to start
        await asyncio.sleep(5)

        # debug: check if server started
        ps_result = await self.sandbox_client.execute_command(self.sandbox_id, "ps aux")
        print(f"[DEBUG] Processes: {ps_result}")

        log_result = await self.sandbox_client.execute_command(self.sandbox_id, "cat /tmp/mcp.log 2>/dev/null || echo 'no log'")
        print(f"[DEBUG] MCP log: {log_result}")

        curl_result = await self.sandbox_client.execute_command(self.sandbox_id, "curl -s http://localhost:3000/mcp 2>&1 || echo 'curl failed'")
        print(f"[DEBUG] Curl localhost: {curl_result}")

        return await super().connect()

    async def expose_sandbox(self) -> None:
        exposed_sandbox = await self.sandbox_client.expose(
            self.sandbox_id,
            self.port_to_expose
        )
        # dns for port exposure takes some time so we wait
        await asyncio.sleep(10)
        self.port_exposed = True
        self.url = f"{exposed_sandbox.url}/mcp"
        self.exposure_id = exposed_sandbox.exposure_id


    async def disconnect(self) -> None:
        await super().disconnect()

        if self.sandbox_id:
            await self.sandbox_client.delete(self.sandbox_id)
            self.sandbox_id = None


