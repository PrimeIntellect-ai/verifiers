"""Dataclasses used to launch MCP stdio servers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(slots=True)
class MCPServerConfig:
    """Simple description of an MCP stdio server."""

    name: str
    command: str
    args: List[str] | None = None
    env: Dict[str, str] | None = None
    description: str = ""
