"""Shared tool contract for composable tasksets and harnesses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TaskTools"]


@dataclass
class TaskTools:
    """Task-provided MCP tools after sandbox-side preparation."""

    mcp_servers: list[dict[str, Any] | str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)

    @property
    def has_harness_tools(self) -> bool:
        return bool(self.mcp_servers)
