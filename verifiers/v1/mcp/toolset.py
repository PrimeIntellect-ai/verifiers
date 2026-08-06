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


class SharedToolsetConfig(BaseConfig):
    runtime: RuntimeConfig = SubprocessConfig()
    url: str | None = None


class Toolset(ServerBase[ConfigT, StateT]):
    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(
            getattr(fn, "tool_name", None) or fn.__name__
            for fn in discover_decorated(self, "tool")
        )

    def _register(self, mcp: FastMCP) -> None:
        for fn, name in zip(
            discover_decorated(self, "tool"), self.tool_names, strict=True
        ):
            mcp.add_tool(
                self._with_state(fn),
                name=name,
                description=(fn.__doc__ or "").strip() or None,
            )
