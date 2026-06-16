"""`ServerBase`: the vf-native server base — a class authored from a config, shared by `Toolset`
and `User`.

A server's env module is self-runnable: its `__main__` calls `ServerBase.run()`, which rebuilds the
server from the environment the framework set and serves it over MCP (`_serve`). The host side that
starts these in a runtime lives in `launch`.
"""

from __future__ import annotations

import functools
import inspect
import os
from typing import TYPE_CHECKING, Callable, ClassVar, Generic, TypeVar, get_args

from pydantic_config import BaseConfig

from verifiers.v1.state import StateT, state_cls

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

ConfigT = TypeVar("ConfigT", bound=BaseConfig)

STATE_TIMEOUT = 30.0
"""Seconds for a state-channel GET/PUT (localhost, or a tunnel to the host)."""


def _import_ref(ref: str) -> object:
    """Resolve a `module:qualname` reference (e.g. `glossary_v1:GlossaryTask`) to the object."""
    import importlib

    module_name, _, qualname = ref.partition(":")
    obj: object = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


class ServerBase(Generic[ConfigT, StateT]):
    """A vf-native server authored as a class, initialized from its config — the same shape as
    `Taskset`/`TasksetConfig`: the config (a `ToolsetConfig`/`UserConfig` subclass) is the
    serializable data (placement + the server's own knobs); the class is the behaviour. The
    framework launches it by running its env module (`python -m <module>`), whose `__main__` calls
    `cls.run()` to rebuild `cls(config)` from the environment and serve over MCP — no FastMCP
    boilerplate. Subclassed by `Toolset` (`@tool` methods) and `User` (a `respond` hook). Build
    expensive/non-serializable state in `setup` — set it as plain instance attributes (it runs in
    the server process). The server's deps come from its env package's `pyproject` (the install in
    a sandbox), so the class may freely `import verifiers`, import siblings, and use module
    globals.

    `self.state` is the rollout's shared `State` (see `verifiers.v1.state`): a `@vf.tool` / `respond`
    reads+writes it, and each call is bracketed (`_with_state`) to pull the latest from the
    interception server and push back any change — so tools and the user sim share state, and a
    server can end the trajectory by setting `self.state.done = True`. Parameterize a stateful server
    with its `State` subclass (`Toolset[Config, MyState]`); `StateT` defaults to the base `State`."""

    TOOL_PREFIX: ClassVar[str] = ""
    """Prefix the model sees on this server's tools (`<TOOL_PREFIX>_<tool>`), set on the class,
    not the config. Empty falls back to the class name snake-cased — set it explicitly for a
    toolset the model calls (e.g. `wiki` -> `wiki_search`)."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self.state: StateT = state_cls(type(self))()
        """The rollout's shared runtime state, refreshed from the channel before each tool/respond
        call (see `_with_state`). Outside a rollout it's just a fresh, inert `State`."""

    def _state_channel(self) -> tuple[str | None, str]:
        """The interception server's state channel `(url, secret)` the framework injected for this
        rollout — `(None, "")` when the server runs outside a rollout (a manual debug run)."""
        return os.environ.get("VF_STATE_URL"), os.environ.get("VF_STATE_SECRET", "")

    async def _pull_state(self) -> dict:
        """Refresh `self.state` from the shared channel (so it reflects other servers' writes) and
        return its JSON snapshot for change detection. Without a channel, reset to a fresh state."""
        url, secret = self._state_channel()
        cls = type(self.state)
        if not url:
            self.state = cls()
            return self.state.model_dump(mode="json")
        import httpx

        async with httpx.AsyncClient(timeout=STATE_TIMEOUT) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {secret}"})
            resp.raise_for_status()
            self.state = cls.model_validate(resp.json())
        return self.state.model_dump(mode="json")

    async def _push_state(self, before: dict) -> None:
        """Push `self.state` back to the shared channel if it changed this call. No-op without a
        channel or when nothing changed."""
        url, secret = self._state_channel()
        if not url:
            return
        after = self.state.model_dump(mode="json")
        if after == before:
            return
        import httpx

        async with httpx.AsyncClient(timeout=STATE_TIMEOUT) as client:
            resp = await client.put(
                url, json=after, headers={"Authorization": f"Bearer {secret}"}
            )
            resp.raise_for_status()

    def _with_state(self, fn: Callable) -> Callable:
        """Wrap a tool/respond callable so each invocation pulls the latest shared `self.state`
        before running and pushes back any change after — the read/write channel a `@vf.tool` and
        `respond` use via `self.state`. Preserves `fn`'s signature so FastMCP advertises the tool
        unchanged. A no-op (fresh inert state) when the server runs outside a rollout (no channel)."""

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            before = await self._pull_state()
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            await self._push_state(before)
            return result

        wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return wrapper

    @property
    def server_name(self) -> str:
        """The server's identity (MCP name, log + namespace key): `TOOL_PREFIX`, else the class
        name snake-cased."""
        return self.TOOL_PREFIX or "".join(
            ("_" + c.lower() if c.isupper() else c) for c in type(self).__name__
        ).lstrip("_")

    async def setup(self) -> None:
        """Task-agnostic setup, in the server process: global state (a corpus / index / graph loaded
        from disk or a dataset) as plain instance attributes (`self.x = ...`). Runs for every server
        — shared or per-rollout. Config knobs stay on `self.config`."""

    async def setup_task(self, task) -> None:
        """Per-rollout setup, in the server process: per-task input read off `task` (this rollout's
        task) and initial per-rollout mutable state (counters, paths). Runs only when the server has
        a task — SKIPPED for a `shared` server (one instance for the whole eval), so don't override
        it on a shared server (the framework warns loudly if you do)."""

    def _register(self, mcp: FastMCP) -> None:
        raise NotImplementedError

    def _serve(self, task) -> None:
        """Run this server's MCP server: bind its port, `setup` (always) + `setup_task(task)` (only
        when there's a task — skipped for a shared server), build a FastMCP from the registered tools,
        and serve it over streamable HTTP on `MCP_HOST`. Called by `run()`."""
        import asyncio
        import socket
        from pathlib import Path

        import uvicorn
        from mcp.server.fastmcp import FastMCP

        host = os.environ.get("MCP_HOST", "127.0.0.1")
        # Bind our own socket up front: `MCP_PORT` when the framework fixed one (a self-publishing
        # runtime's forwarded port), else 0 = an OS-assigned free port — guaranteed free in whatever
        # environment we run in (host or sandbox), so the launcher never probes for a free port.
        # Report the bound port back via `MCP_PORT_FILE` before setup, so the launcher learns it
        # without waiting on a slow `setup` (its readiness probe absorbs that).
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(os.environ.get("MCP_PORT", 0))))
        port_file = os.environ.get("MCP_PORT_FILE")
        if port_file:
            Path(port_file).write_text(str(sock.getsockname()[1]))

        async def _setup() -> None:
            await self.setup()
            if task is not None:
                await self.setup_task(task)

        asyncio.run(_setup())
        # Bound to 0.0.0.0 means a sandbox tunnel reaches us at a non-loopback host (e.g. modal's
        # *.modal.host); FastMCP's default DNS-rebinding guard allows only localhost and would 421
        # the tunnel host, so relax it (the tunnel is the trust boundary).
        security = None
        if host == "0.0.0.0":
            from mcp.server.transport_security import TransportSecuritySettings

            security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        mcp = FastMCP(self.server_name, transport_security=security)
        self._register(mcp)
        server = uvicorn.Server(
            uvicorn.Config(mcp.streamable_http_app(), log_level="critical")
        )
        asyncio.run(server.serve(sockets=[sock]))

    @classmethod
    def _config_cls(cls) -> type[BaseConfig]:
        """The config type from the `Toolset[Config]` / `User[Config]` generic parameter. Walks the
        MRO, so a further subclass that doesn't re-parameterize (`class B(MyToolset)`) inherits the
        config from where it was set."""
        for klass in cls.__mro__:
            for base in getattr(klass, "__orig_bases__", ()):
                for arg in get_args(base):
                    if isinstance(arg, type) and issubclass(arg, BaseConfig):
                        return arg
        raise TypeError(
            f"{cls.__name__} must parameterize its config, e.g. Toolset[MyConfig]"
        )

    @classmethod
    def run(cls) -> None:
        """Entry point a server module calls from `if __name__ == "__main__"`: rebuild this server
        from the environment the framework set (`VF_CONFIG` JSON + `VF_TASK`/`VF_TASK_CLS`) and serve
        it over MCP. With no `VF_CONFIG` the config is parsed from the CLI instead (`cli(config)`),
        so the module is runnable by hand for debugging. `VF_TASK` is absent for a `shared` server."""
        config_cls = cls._config_cls()
        if "VF_CONFIG" in os.environ:
            config = config_cls.model_validate_json(os.environ["VF_CONFIG"])
        else:
            from pydantic_config import cli

            config = cli(config_cls)
        task = None
        if "VF_TASK" in os.environ:
            task_cls = os.environ.get("VF_TASK_CLS")
            if task_cls is None:
                raise ValueError(
                    "VF_TASK is set but VF_TASK_CLS is not; the framework sets both together "
                    "(VF_TASK_CLS names the Task subclass to rebuild the task with)"
                )
            task = _import_ref(task_cls).model_validate_json(os.environ["VF_TASK"])
        cls(config)._serve(task)
