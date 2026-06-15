"""The dependency-light runtime that serves a vf-native `Toolset`/`User` over MCP.

This module's *source* is vendored (inlined) into the PEP 723 uv-script the framework renders for
each server — it is never imported as `verifiers.v1._serverkit` at serve time. So the rendered
script is standalone: it depends only on `mcp` + `pydantic` + `uvicorn` (+ the class's own declared
`deps`) — all public PyPI — never `verifiers` or the taskset package. The API here mirrors the
server-facing slice of `verifiers.v1`, so the authored class (which uses `vf.Toolset` / `@vf.tool`
/ `vf.UserConfig` …) runs unchanged against it once `vf` is bound to the inlined runtime.

The boundary contract a server must honour to render: it may only touch this runtime, its config,
the `task`, and its declared `deps` — no taskset module globals or sibling imports. State that
crosses the wire is its `config` (JSON); state built in the server process lives in `setup`.
"""

import inspect
import os
from typing import Any, Callable, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

F = TypeVar("F", bound=Callable[..., Any])
ConfigT = TypeVar("ConfigT", bound="ToolsetConfig")

Message = dict[str, Any]
Messages = list[Message]


def mark(attr: str, **extra: Any) -> Callable[[F], F]:
    def decorator(f: F) -> F:
        setattr(f, attr, True)
        for key, value in extra.items():
            setattr(f, key, value)
        return f

    return decorator


def discover_decorated(obj: object, attr: str) -> list[Callable[..., Any]]:
    """Bound methods on `obj` tagged with `attr`, sorted by name."""
    methods = [
        m for _, m in inspect.getmembers(obj, predicate=inspect.ismethod) if hasattr(m, attr)
    ]
    methods.sort(key=lambda m: m.__name__)
    return methods


def tool(func: F | None = None, name: str | None = None) -> F | Callable[[F], F]:
    """Mark a `Toolset` method as an MCP tool (name defaults to the method; docstring → description)."""
    decorator = mark("tool", tool_name=name)
    return decorator if func is None else decorator(func)


class Task(BaseModel):
    """A permissive view of the rollout's task — whatever fields the taskset put on it (`info`,
    `source`, `target`, …), read by `setup`. Not the taskset's typed `Task`, so the server needs
    no framework import."""

    model_config = ConfigDict(extra="allow")


class ToolsetConfig(BaseModel):
    """Server-side base config: holds the toolset's own knobs (subclass to add them). Placement
    fields the host carries (colocated / shared / runtime / url) are irrelevant here and ignored."""

    model_config = ConfigDict(extra="ignore")


class UserConfig(BaseModel):
    """Server-side base config for a user simulator (see `ToolsetConfig`)."""

    model_config = ConfigDict(extra="ignore")


class ServerBase(Generic[ConfigT]):
    """A vf-native server initialized from its config. Subclassed by `Toolset` / `User`."""

    name: ClassVar[str] = ""
    deps: ClassVar[list[str]] = []

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    @property
    def server_name(self) -> str:
        return self.name or "".join(
            ("_" + c.lower() if c.isupper() else c) for c in type(self).__name__
        ).lstrip("_")

    async def setup(self, task: Task) -> None:
        """Build global / per-task / mutable state in the server process, as plain attributes."""

    def _register(self, mcp: Any) -> None:
        raise NotImplementedError


class Toolset(ServerBase[ConfigT]):
    """A tool server: `@tool` methods become MCP tools `<name>_<method>`."""

    def _register(self, mcp: Any) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                fn,
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )


class User(ServerBase[ConfigT]):
    """A user simulator: a single `respond(message) -> (messages, done)` hook the framework drives."""

    async def respond(self, message: str) -> tuple[Messages, bool]:
        raise NotImplementedError

    def _register(self, mcp: Any) -> None:
        import json

        user = self

        async def respond(message: str) -> str:
            messages, done = await user.respond(message)
            wire = [m if isinstance(m, dict) else m.model_dump(exclude_none=True) for m in messages]
            return json.dumps({"messages": wire, "done": done})

        mcp.add_tool(respond, name="respond")


def run_mcp_server(mcp: Any) -> None:
    """Serve a FastMCP on the port the harness passes via `MCP_PORT`, streamable HTTP at `/mcp`."""
    import uvicorn

    port = int(os.environ["MCP_PORT"])
    config = uvicorn.Config(
        mcp.streamable_http_app(), host="127.0.0.1", port=port, log_level="critical"
    )
    uvicorn.Server(config).run()


def serve_server(inst: ServerBase, task: Task) -> None:
    """Await `inst.setup(task)`, build a FastMCP from its registered tools, and serve it."""
    import asyncio

    from mcp.server.fastmcp import FastMCP

    asyncio.run(inst.setup(task))
    mcp = FastMCP(inst.server_name)
    inst._register(mcp)
    run_mcp_server(mcp)
