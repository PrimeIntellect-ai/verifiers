from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.mcp.server import ConfigT, ServerBase
from verifiers.v1.runtimes import RuntimeConfig, SubprocessConfig
from verifiers.v1.state import StateT
from verifiers.v1.utils.decorators import discover_decorated

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer


class ToolsetTimeoutConfig(BaseConfig):
    """Timeouts (in seconds) for a launched tool server; `None` waits indefinitely."""

    startup: float | None = Field(600.0, gt=0)
    """Time to start serving, including `setup()`, such as to build an index."""


class BaseToolsetConfig(BaseConfig):
    """Settings every tool server has, whether per-task or shared."""

    runtime: RuntimeConfig = SubprocessConfig()
    """Where the server runs: the evaluator by default."""
    url: str | None = None
    timeout: ToolsetTimeoutConfig = ToolsetTimeoutConfig()


class ToolsetConfig(BaseToolsetConfig):
    """A per-task tool server. A Prime `runtime` beside a Prime workspace without a task
    service spec is a VM of the server's own that joins the workspace over the rollout's
    private network, under its own egress policy."""

    colocated: bool = False
    service: str | None = None
    """Run the server in a named native Prime service VM. The harness reaches it
    over the deployment's private network, using its LAN hostname."""

    @model_validator(mode="after")
    def _one_placement(self):
        if self.service is None:
            return self
        if self.colocated or self.url:
            raise ValueError(
                "a tool server placed in a service is neither colocated nor remote"
            )
        # The service is the server's runtime; another one, or its policy, would go unused.
        if not isinstance(self.runtime, SubprocessConfig):
            raise ValueError(  # noqa: TRY004 - Pydantic validators must raise ValueError
                "a tool server placed in a service runs in that service and "
                f"takes no runtime of its own (got {self.runtime.type})"
            )
        return self


class SharedToolsetConfig(BaseToolsetConfig):
    pass


class Toolset(ServerBase[ConfigT, StateT]):
    def register(self, mcp: MCPServer) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )
