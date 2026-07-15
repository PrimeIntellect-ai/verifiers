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
    scratch_dir: str = "/tmp"
    """Where a sandbox-launched server keeps its venv and uv caches. Point it at a disk-backed
    writable path when the sandbox's /tmp is a small tmpfs (e.g. prime VM-image sandboxes)."""


class SharedToolsetConfig(BaseConfig):
    runtime: RuntimeConfig = SubprocessConfig()
    url: str | None = None
    scratch_dir: str = "/tmp"
    """See `ToolsetConfig.scratch_dir`."""


class Toolset(ServerBase[ConfigT, StateT]):
    def _register(self, mcp: FastMCP) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                self._with_state(fn),
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )
