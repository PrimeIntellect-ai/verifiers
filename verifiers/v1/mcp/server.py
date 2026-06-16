"""`ServerBase`: the vf-native server base — a class authored from a config, shared by `Toolset`
and `User`.

A server's env module is self-runnable: its `__main__` calls `ServerBase.run()`, which rebuilds the
server from the environment the framework set and serves it over MCP (`_serve`). The host side that
starts these in a runtime lives in `launch`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, get_args

from pydantic_config import BaseConfig

from verifiers.v1.runtimes.base import SERVICE_PORT

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

ConfigT = TypeVar("ConfigT", bound=BaseConfig)


def _import_ref(ref: str) -> object:
    """Resolve a `module:qualname` reference (e.g. `glossary_v1:GlossaryTask`) to the object."""
    import importlib

    module_name, _, qualname = ref.partition(":")
    obj: object = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


class ServerBase(Generic[ConfigT]):
    """A vf-native server authored as a class, initialized from its config — the same shape as
    `Taskset`/`TasksetConfig`: the config (a `ToolsetConfig`/`UserConfig` subclass) is the
    serializable data (placement + the server's own knobs); the class is the behaviour. The
    framework launches it by running its env module (`python -m <module>`), whose `__main__` calls
    `cls.run()` to rebuild `cls(config)` from the environment and serve over MCP — no FastMCP
    boilerplate. Subclassed by `Toolset` (`@tool` methods) and `User` (a `respond` hook). Build
    expensive/non-serializable state in `setup` — set it as plain instance attributes (it runs in
    the server process). The server's deps come from its env package's `pyproject` (the install in
    a sandbox), so the class may freely `import verifiers`, import siblings, and use module
    globals."""

    TOOL_PREFIX: ClassVar[str] = ""
    """Prefix the model sees on this server's tools (`<TOOL_PREFIX>_<tool>`), set on the class,
    not the config. Empty falls back to the class name snake-cased — set it explicitly for a
    toolset the model calls (e.g. `wiki` -> `wiki_search`)."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

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
        """Run this server's MCP server: `setup` (always) + `setup_task(task)` (only when there's a
        task — skipped for a shared server), build a FastMCP from the registered tools, and serve it
        over streamable HTTP on `MCP_PORT`/`MCP_HOST`. Called by `run()`."""
        import asyncio

        import uvicorn
        from mcp.server.fastmcp import FastMCP

        async def _setup() -> None:
            await self.setup()
            if task is not None:
                await self.setup_task(task)

        asyncio.run(_setup())
        # Bound to 0.0.0.0 means a sandbox tunnel reaches us at a non-loopback host (e.g. modal's
        # *.modal.host); FastMCP's default DNS-rebinding guard allows only localhost and would 421
        # the tunnel host, so relax it (the tunnel is the trust boundary).
        host = os.environ.get("MCP_HOST", "127.0.0.1")
        security = None
        if host == "0.0.0.0":
            from mcp.server.transport_security import TransportSecuritySettings

            security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        mcp = FastMCP(self.server_name, transport_security=security)
        self._register(mcp)
        uvicorn.Server(
            uvicorn.Config(
                mcp.streamable_http_app(),
                host=host,
                port=int(os.environ.get("MCP_PORT", SERVICE_PORT)),
                log_level="critical",
            )
        ).run()

    @classmethod
    def _config_cls(cls) -> type[BaseConfig]:
        """The config type from the `Toolset[Config]` / `User[Config]` generic parameter."""
        for base in getattr(cls, "__orig_bases__", ()):
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
