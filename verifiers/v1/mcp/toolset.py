from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.mcp.server import ConfigT, ServerBase
from verifiers.v1.runtimes import DockerConfig, PrimeConfig, SubprocessConfig
from verifiers.v1.state import StateT
from verifiers.v1.utils.decorators import discover_decorated

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer


class ToolsetTimeoutConfig(BaseConfig):
    """Timeouts (in seconds) for a launched tool server; `None` waits indefinitely."""

    startup: float | None = Field(600.0, gt=0)
    """Time to start serving, including `setup()`, such as to build an index."""


class ToolsetConfig(BaseConfig):
    """Where a tool server runs and how long it lives.

    A subprocess or Docker `runtime` serves on the evaluator's loopback; the harness
    reaches it directly from a local workspace and through its interception from a
    Prime one. A Prime `runtime` is a VM of the server's own that joins a Prime
    workspace over a private LAN, under its own egress policy."""

    shared: bool = False
    """Serve every rollout of an eval from one server instead of starting one per
    rollout. A shared server runs `setup()` once and never `setup_task()`."""
    service: str | None = None
    """Run the server in one of the task's services instead of a runtime of its own.
    The harness reaches it by the service's name on the deployment's network."""
    runtime: SerializeAsAny[
        Annotated[
            SubprocessConfig | DockerConfig | PrimeConfig, Field(discriminator="type")
        ]
    ] = SubprocessConfig()
    url: str | None = None
    """An MCP server that is already running; nothing is launched."""
    timeout: ToolsetTimeoutConfig = ToolsetTimeoutConfig()

    @model_validator(mode="after")
    def _one_placement(self):
        if self.service is not None:
            if self.url or self.shared:
                raise ValueError(
                    "a tool server placed in a service belongs to one rollout and "
                    "is not remote"
                )
            # The service is the server's runtime; another one, or its policy, would go unused.
            if not isinstance(self.runtime, SubprocessConfig):
                raise ValueError(
                    "a tool server placed in a service runs in that service and "
                    f"takes no runtime of its own (got {self.runtime.type})"
                )
        # Only a rollout's own Prime VM applies a network policy of its own, after setup.
        restricted = (
            not isinstance(self.runtime, SubprocessConfig)
            and self.runtime.network_restricted
        )
        if restricted and (isinstance(self.runtime, DockerConfig) or self.shared):
            raise ValueError(
                "a tool server's own network policy needs a Prime VM that serves "
                "one rollout"
            )
        return self


class Toolset(ServerBase[ConfigT, StateT]):
    def register(self, mcp: MCPServer) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )
