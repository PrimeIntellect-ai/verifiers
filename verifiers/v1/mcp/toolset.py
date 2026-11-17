from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.mcp.server import ConfigT, ServerBase
from verifiers.v1.runtimes import RuntimeConfig
from verifiers.v1.state import StateT
from verifiers.v1.utils.decorators import discover_decorated

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

    from verifiers.v1.runtimes import Runtime


class ToolsetTimeoutConfig(BaseConfig):
    """Timeouts (in seconds) for a launched tool server; `None` waits indefinitely."""

    startup: float | None = Field(600.0, gt=0)
    """Time to start serving, including `setup()`, such as to build an index."""


class BaseToolsetConfig(BaseConfig):
    """Runtime and connection settings for per-task and shared tool servers."""

    runtime: RuntimeConfig | None = None
    """Where the server runs. None runs it in a sandbox of its own from the harness's
    provider: a Docker container beside a Docker harness, a Prime VM beside a Prime
    one. A server on a local runtime cannot serve a remote harness."""
    url: str | None = None
    timeout: ToolsetTimeoutConfig = ToolsetTimeoutConfig()


class ToolsetConfig(BaseToolsetConfig):
    """A per-task tool server. On Prime beside a Prime workspace without a task service
    spec, its VM joins the workspace over the rollout's private network, under its own
    egress policy."""

    service: str | None = None
    """Run the server in a named native Prime service VM. The harness reaches it
    over the deployment's private network, using its LAN hostname."""

    @model_validator(mode="after")
    def _one_placement(self):
        if self.service is None:
            return self
        if self.url:
            raise ValueError(
                "a tool server placed in a service cannot use a remote URL"
            )
        # The service is the server's runtime; another one, or its policy, would go unused.
        if self.runtime is not None:
            raise ValueError(
                "a tool server placed in a service runs in that service and "
                f"takes no runtime of its own (got {self.runtime.type})"
            )
        return self


class SharedToolsetConfig(BaseToolsetConfig):
    pass


class Toolset(ServerBase[ConfigT, StateT]):
    @asynccontextmanager
    async def serve(self, runtime: Runtime, **kwargs) -> AsyncIterator[str]:
        """Connect a rollout to this server; task-format adapters own their transport."""
        from verifiers.v1.mcp.launch import serve

        if self.config.url:
            yield runtime.host_url(self.config.url)
        else:
            async with serve(self, runtime, **kwargs) as url:
                yield url

    def register(self, mcp: MCPServer) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )
