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
    on: str | None = None
    """The node the server runs on: `"host"` (the evaluator), `"main"` (the
    workspace), a node the task declares (`TaskData.nodes`), or a service of its
    Compose project. The harness reaches a server on another node over the rollout's
    network, by that node's name, so the workspace needs no access to whatever the
    server fronts. Unset, `colocated`, `service` and `runtime` decide."""
    colocated: bool = False
    """Run the server in the workspace: `on = "main"`."""
    runtime: RuntimeConfig = SubprocessConfig()
    """Where the server runs when `on` is unset: the evaluator by default. A runtime of
    the workspace's type, beside a workspace without Compose services whose runtime
    joins machines on a private network (Prime), is a machine of the server's own on
    that network, under its own egress policy."""
    url: str | None = None
    service: str | None = None
    """Run the server in this service of the task's Compose services: `on = service`."""

    @property
    def node(self) -> str | None:
        """The node `on`, `colocated` or `service` names, if any."""
        if self.on is not None:
            return self.on
        return "main" if self.colocated else self.service

    @model_validator(mode="after")
    def _one_placement(self):
        if self.on is not None and (self.colocated or self.service or self.url):
            raise ValueError(
                "a tool server placed with `on` takes no `colocated`, `service` or `url`"
            )
        if self.service is not None and (self.colocated or self.url):
            raise ValueError(
                "a tool server placed in a Compose service is neither colocated nor remote"
            )
        # The node is the server's runtime; another one, or its policy, would go unused.
        if self.node not in (None, "host", "main") and not isinstance(
            self.runtime, SubprocessConfig
        ):
            raise ValueError(
                f"a tool server on node {self.node!r} runs on that node and takes no "
                f"runtime of its own (got {self.runtime.type})"
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
