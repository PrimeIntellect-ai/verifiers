from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_config import BaseConfig

from verifiers.v1.mcp.server import ConfigT, ServerBase
from verifiers.v1.runtimes import RuntimeConfig, SubprocessConfig
from verifiers.v1.state import StateT
from verifiers.v1.utils.decorators import discover_decorated

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class ToolsetConfig(BaseConfig):
    colocated: bool = False
    runtime: RuntimeConfig = SubprocessConfig()
    url: str | None = None
    install_dir: str | None = None
    """Where to build the server's venv in a sandboxed runtime (default: under /tmp).
    Point this at a path on the real disk when the image caps /tmp — prime VM boxes mount
    it as a 485 MB tmpfs, too small for a server whose dependency closure is large."""


class SharedToolsetConfig(BaseConfig):
    runtime: RuntimeConfig = SubprocessConfig()
    url: str | None = None
    install_dir: str | None = None
    """Where to build the server's venv in a sandboxed runtime (see `ToolsetConfig`)."""


class Toolset(ServerBase[ConfigT, StateT]):
    def _register(self, mcp: FastMCP) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )
