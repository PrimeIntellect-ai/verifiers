"""vf-native MCP servers: tool servers (`Toolset`) and user simulators (`User`).

A server is a class authored from a config (`server.ServerBase`), self-launched via its env module's
`__main__` (`ServerBase.run`). The host side (`launch`) brings servers up in a runtime and reaches
them: `serve` (one server, any placement), `serve_tools` / `serve_user` (a rollout's tools /
the user sim), `SharedServers` (the run-scoped lazy registry of `shared` servers), and
`connect_user` (the MCP client the framework drives the user sim through).
"""

from verifiers.v1.mcp.launch import (
    Respond,
    SharedServers,
    connect_user,
    serve,
    serve_tools,
    serve_user,
)
from verifiers.v1.mcp.server import ServerBase
from verifiers.v1.mcp.toolset import Toolset, ToolsetConfig
from verifiers.v1.mcp.user import User, UserConfig

__all__ = [
    "ServerBase",
    "Toolset",
    "ToolsetConfig",
    "User",
    "UserConfig",
    "Respond",
    "SharedServers",
    "connect_user",
    "serve",
    "serve_tools",
    "serve_user",
]
