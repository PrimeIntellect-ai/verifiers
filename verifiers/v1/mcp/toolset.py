from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import model_validator
from pydantic_config import BaseConfig

from verifiers.v1.mcp.server import ConfigT, ServerBase
from verifiers.v1.runtimes import RuntimeConfig, SubprocessConfig
from verifiers.v1.state import StateT
from verifiers.v1.utils.decorators import discover_decorated

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer


class ToolsetConfig(BaseConfig):
    colocated: bool = False
    runtime: RuntimeConfig = SubprocessConfig()
    """Where the server runs: the evaluator by default. A Prime runtime beside a Prime
    workspace without Compose services is a VM of the server's own that joins the
    workspace over the rollout's private network, under its own egress policy."""
    url: str | None = None
    service: str | None = None
    """Run the server in this service of the task's Compose services (on a Prime
    VM). The harness reaches it over the Compose network by the service's name, so the
    workspace needs no access to whatever the server fronts."""

    @model_validator(mode="after")
    def _one_placement(self):
        if self.service is None:
            return self
        if self.colocated or self.url:
            raise ValueError(
                "a tool server placed in a Compose service is neither colocated nor remote"
            )
        # The service is the server's runtime; another one, or its policy, would go unused.
        if not isinstance(self.runtime, SubprocessConfig):
            raise ValueError(  # noqa: TRY004 - Pydantic validators must raise ValueError
                "a tool server placed in a Compose service runs in that service and "
                f"takes no runtime of its own (got {self.runtime.type})"
            )
        return self


class SharedToolsetConfig(BaseConfig):
    runtime: RuntimeConfig = SubprocessConfig()
    url: str | None = None


class Toolset(ServerBase[ConfigT, StateT]):
    def register(self, mcp: MCPServer) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )
