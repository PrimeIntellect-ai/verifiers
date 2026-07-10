from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_config import BaseConfig

from verifiers.v1.decorators import discover_decorated
from verifiers.v1.mcp.server import ConfigT, ServerBase
from verifiers.v1.runtimes import RuntimeConfig, SubprocessConfig
from verifiers.v1.state import StateT

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class ToolsetConfig(BaseConfig):
    colocated: bool = False
    runtime: RuntimeConfig = SubprocessConfig()
    url: str | None = None


class SharedToolsetConfig(BaseConfig):
    runtime: RuntimeConfig = SubprocessConfig()
    url: str | None = None


class Toolset(ServerBase[ConfigT, StateT]):
    def _register(self, mcp: FastMCP) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )
