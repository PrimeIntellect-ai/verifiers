"""Tools: how a task gives the harness tools, via tool servers it declares.

A taskset returns `Tools`s from `Taskset.tools`. A server is either a
single-file uv script the harness runs (`script` — its only runtime dep is `uv`, which
resolves the script's PEP 723 inline deps, so it runs in any runtime: host, the harness's
runtime when colocated, or its own) or an already-running remote endpoint (`url`, e.g.
deepwiki). The server binds the port passed via `MCP_PORT`.

`serve_tools` brings a task's servers up for a rollout — colocated in the harness's runtime
(reached via localhost) or in their own `tools.runtime` (reached via that runtime's
`public_url`, or the harness runtime's `expose` for a tunnel) — and yields `{name: url}`;
`serve_shared` brings up shared ones once per eval. `run_mcp_server` is the server-side
launcher. The wire types the model sees (`Tool`, `ToolCall`, …) live in `types`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import socket
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.decorators import discover_decorated
from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes import Runtime, RuntimeConfig, SubprocessConfig, make_runtime
from verifiers.v1.runtimes.base import _ENSURE_UV

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


class ToolsetConfig(BaseConfig):
    """Where one tool server runs (placement). The default — its own host (`subprocess`)
    runtime — is the robust one: the server runs in the eval process's environment, where
    `verifiers` and the taskset package are already installed, and the harness reaches it over
    the host network (docker `--network host`) or a tunnel (prime), so the sandbox needs nothing
    installed. `colocated` and `shared` trade that off:
      - neither (default): its own `runtime`, per rollout (host by default).
      - colocated: in the harness's OWN runtime, per rollout (no tunnel) — only works when the
        server's deps resolve there (a self-contained published script), not a bare sandbox.
      - shared: one instance for the whole eval, in its own `runtime`."""

    colocated: bool = False
    """Run the server inside the harness's runtime (reached in-sandbox, no tunnel). Off by
    default: a vf-native server imports `verifiers` + its taskset package, which a fresh
    sandbox doesn't have — so it runs host-side instead. Turn on only for a self-contained
    server whose deps resolve inside the harness runtime."""
    shared: bool = False
    """Run one server instance for the whole eval, shared across rollouts (in its own
    `runtime`). Mutually exclusive with `colocated`."""
    runtime: RuntimeConfig = SubprocessConfig()
    """The server's own runtime, used unless `colocated` (host/subprocess by default — always
    reachable from any harness runtime; set docker/prime to isolate it in its own sandbox)."""

    @model_validator(mode="after")
    def _exclusive(self) -> "ToolsetConfig":
        if self.colocated and self.shared:
            raise ValueError("colocated and shared are mutually exclusive")
        return self


class UserConfig(BaseConfig):
    """Where the user simulator runs (placement). The framework always drives it from the host.
    Default — its own host (`subprocess`) runtime — runs it where `verifiers` + the taskset
    package live, reachable from any harness runtime, so the sandbox needs nothing. Set
    `colocated` to run it inside the harness's runtime instead (only when its deps resolve
    there)."""

    colocated: bool = False
    """Run the user simulator inside the harness's runtime, reusing it (its port is published
    back to the host so the framework can still drive it). Off by default — see `ToolsetConfig`."""
    runtime: RuntimeConfig = SubprocessConfig()
    """The user simulator's own runtime, used unless `colocated` (host/subprocess by default)."""


@dataclass(frozen=True)
class Tools:
    """A tool server exposing tools to the model (as `<name>_<tool>`). Set exactly one
    of `script` (a single-file uv script we run) or `url` (already-running remote)."""

    name: str
    script: bytes | None = None
    """A self-contained uv script (PEP 723 inline deps) that serves streamable HTTP on
    the port in `MCP_PORT` — e.g. via `run_mcp_server`. Its only runtime dep is `uv`."""
    command: list[str] = field(default_factory=list)
    """Alternative to `script`: a ready-to-run argv that serves on `MCP_PORT` (for a
    server whose deps are already present where it runs)."""
    env: dict[str, str] = field(default_factory=dict)
    """Extra env for the script/command process (e.g. per-task metadata)."""
    url: str | None = None
    """An already-running streamable-HTTP MCP endpoint the harness connects to."""
    headers: dict[str, str] = field(default_factory=dict)
    """HTTP headers sent to a remote `url` (e.g. auth)."""
    config: ToolsetConfig = field(default_factory=ToolsetConfig)
    """Per-server placement (colocated / shared / own runtime)."""


def _server_name(inst: ServerBase) -> str:
    """The MCP server name: the instance's `name`, else the class name snake-cased."""
    if inst.name:
        return inst.name
    return "".join(
        ("_" + c.lower() if c.isupper() else c) for c in type(inst).__name__
    ).lstrip("_")


class ServerBase(BaseModel):
    """A vf-native server authored as a class: config is pydantic fields, behaviour is methods.
    The framework serializes it, launches `verifiers.v1.toolserver` in a runtime, rebuilds it
    there, and serves it over MCP — no FastMCP boilerplate. Subclassed by `Toolset` (model-facing
    `@tool` methods) and `User` (a single `respond` hook). Declare extra PyPI deps in `deps`
    (class-level, the equivalent of a uv script's PEP 723 deps) so the framework can resolve them
    in any runtime; build expensive/non-serializable state in `setup` (runs in the server proc)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    deps: ClassVar[list[str]] = []
    name: str = ""
    """MCP server name; the model sees tools as `<name>_<tool>`. Defaults to the class name."""
    config: ToolsetConfig = Field(default_factory=ToolsetConfig, exclude=True)
    """Per-server placement (colocated / shared / own runtime). Excluded from the serialized
    payload — placement is resolved host-side, the server process never needs it. `User`
    narrows this to a `UserConfig`."""

    async def setup(self) -> None:
        """Build expensive / non-serializable state (corpus, index, …) in the server process."""

    def _register(self, mcp: FastMCP) -> None:
        raise NotImplementedError


class Toolset(ServerBase):
    """A tool server authored as a class: write `@tool`-decorated methods (the model calls them
    as `<name>_<method>`; the docstring is the tool description). Example:

        class GlossaryToolset(vf.Toolset):
            facts: dict[str, str]
            @vf.tool
            def lookup(self, name: str) -> str: return self.facts.get(name.lower(), "unknown")
    """

    def _register(self, mcp: FastMCP) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                fn,
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )


def serve_server(inst: ServerBase) -> None:
    """Run a `ServerBase` instance's MCP server (called by the `toolserver` launcher): await its
    `setup`, build a FastMCP from its registered tools, and serve via `run_mcp_server`."""
    import asyncio

    from mcp.server.fastmcp import FastMCP

    asyncio.run(inst.setup())
    mcp = FastMCP(_server_name(inst))
    inst._register(mcp)
    run_mcp_server(mcp)


# A generated PEP 723 uv-script that rebuilds the server class in any runtime: uv resolves the
# class's deps (+ verifiers + the taskset package), then `serve_server` runs it.
_SERVER_SCRIPT = '''\
# /// script
# requires-python = ">=3.10"
# dependencies = [{deps}]
# ///
import importlib, os

from verifiers.v1.tools import serve_server

mod, qual = os.environ["VF_SERVER_REF"].split(":")
cls = getattr(importlib.import_module(mod), qual)
serve_server(cls.model_validate_json(os.environ["VF_SERVER"]))
'''


def _provides_top_level(dist, top: str) -> bool:
    """Whether an installed distribution exposes `top` as an importable top-level package —
    via its recorded files / `top_level.txt` (regular installs) or its source tree (editable
    installs, whose RECORD lists only the `.pth`, so we read `direct_url.json`)."""
    import json
    from pathlib import Path

    if top in (dist.read_text("top_level.txt") or "").split():
        return True
    for f in dist.files or []:
        if f.parts and f.parts[0] in (top, f"{top}.py"):
            return True
    raw = dist.read_text("direct_url.json")
    if raw:
        info = json.loads(raw)
        url = info.get("url", "")
        if info.get("dir_info", {}).get("editable") and url.startswith("file://"):
            src = Path(url[len("file://") :])
            return (src / top / "__init__.py").exists() or (src / f"{top}.py").exists()
    return False


def _server_distribution(cls: type) -> str | None:
    """The installed distribution providing `cls`'s top-level import package, so a generated
    sandbox script can install it with `uv`. Handles editable installs (uv / PEP 660) that
    `packages_distributions()` misses. None if no installed distribution provides it (the class
    isn't part of a publishable package)."""
    import importlib.metadata as md

    top = cls.__module__.split(".")[0]
    direct = md.packages_distributions().get(top)
    if direct:
        return direct[0]
    for dist in md.distributions():
        if _provides_top_level(dist, top):
            return dist.metadata["Name"]
    return None


def server_to_tools(inst: ServerBase, runtime_type: str) -> Tools:
    """Convert a vf-native `ServerBase` to a `Tools` launch. On a host subprocess (deps already
    present) run it directly via the `toolserver` launcher as a `command`; in any other runtime
    generate a PEP 723 uv-script so `uv` resolves the declared `deps` (the deps answer) plus the
    class's own distribution (so the launcher can import it) in that fresh sandbox."""
    cls = type(inst)
    env = {
        "VF_SERVER": inst.model_dump_json(),
        "VF_SERVER_REF": f"{cls.__module__}:{cls.__qualname__}",
    }
    name = _server_name(inst)
    if runtime_type == "subprocess":
        return Tools(
            name=name, command=[sys.executable, "-m", "verifiers.v1.toolserver"], env=env
        )
    pkg = _server_distribution(cls)
    if pkg is None:
        raise ProgramError(
            f"cannot run {cls.__qualname__} in a {runtime_type} runtime: it must live in an "
            "installed, publishable distribution so the sandbox can `uv`-install it (it resolved "
            "to no distribution). Put the class in a taskset package, or run it colocated in a "
            "subprocess runtime."
        )
    deps = ["mcp", "verifiers", *cls.deps, pkg]
    script = _SERVER_SCRIPT.format(deps=", ".join(f'"{d}"' for d in deps))
    return Tools(name=name, script=script.encode(), env=env)


def run_mcp_server(mcp: "FastMCP") -> None:
    """Serve a FastMCP server on the port the harness passes via `MCP_PORT`, mounting
    streamable HTTP at `/mcp`. Call this at the end of a uv-script tool server."""
    import uvicorn

    port = int(os.environ["MCP_PORT"])
    config = uvicorn.Config(
        mcp.streamable_http_app(), host="127.0.0.1", port=port, log_level="critical"
    )
    uvicorn.Server(config).run()


# Poll a URL from inside a runtime until it serves (any HTTP response — incl. MCP's 406
# to a bare GET — means up), or the deadline passes. python3 is present in every runtime.
_PROBE = """
import sys, time, urllib.error, urllib.request
for _ in range(180):
    try:
        urllib.request.urlopen(sys.argv[1], timeout=2); sys.exit(0)
    except urllib.error.HTTPError:
        sys.exit(0)
    except Exception:
        time.sleep(1)
sys.exit(1)
"""


def _free_port() -> int:
    """A free host port in [3000, 9000) — also free in a fresh container/sandbox, and
    within prime's port-exposure cap (<= 9000)."""
    for _ in range(50):
        port = random.randint(3000, 8999)
        probe = socket.socket()
        try:
            probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            probe.close()
    raise ProgramError("could not find a free port in [3000, 9000)")


async def serve_in_runtime(server: Tools, runtime: Runtime, port: int) -> None:
    """Run `server` inside `runtime` on `port` (background) and wait until it serves.
    A `script` runs via uv (its deps come from the script); a `command` must already be
    runnable in the runtime."""
    log = f"vf_tool_{server.name}.log"
    if server.script:
        script = f"vf_tool_{server.name}.py"
        await runtime.write(script, server.script)
        argv = ["sh", "-c", f"{_ENSURE_UV}; exec uv run --quiet {script}"]
    else:
        argv = list(server.command)
    await runtime.run_background(argv, {**server.env, "MCP_PORT": str(port)}, log)
    probe = await runtime.run(
        ["python3", "-c", _PROBE, f"http://127.0.0.1:{port}/mcp"], {}
    )
    if probe.exit_code != 0:
        tail = ""
        with contextlib.suppress(Exception):
            tail = (await runtime.read(log)).decode(errors="replace").strip()[-2000:]
        raise ProgramError(
            f"tool server {server.name!r} not serving in runtime: {tail}"
        )


async def _resolve_url(tool_runtime: Runtime, agent_runtime: Runtime, port: int) -> str:
    """A URL the harness can reach the tool's `port` at: the tool runtime publishes it if
    it can (a prime sandbox), else the harness runtime bridges the host port (localhost or
    a tunnel)."""
    base = await tool_runtime.public_url(port) or await agent_runtime.expose(port)
    return f"{base.rstrip('/')}/mcp"


def _descriptor(server: Tools | ServerBase, runtime_type: str) -> Tools:
    """The launch descriptor for a server: a vf-native class is converted for `runtime_type`;
    a raw `Tools` is already one."""
    return server_to_tools(server, runtime_type) if isinstance(server, ServerBase) else server


@contextlib.asynccontextmanager
async def serve_shared(tools: list[Tools | ServerBase]):
    """Start the SHARED tool servers (those whose placement is `shared`) ONCE for a whole eval,
    each in its OWN `runtime`, and yield `{name: url}` reachable by every rollout's harness — a
    prime tool runtime publishes its port (works for any harness), a host one is reached at
    localhost (works for host-network harnesses). Torn down when the eval ends. Used by
    `Environment` so an expensive corpus is built once, not per rollout."""
    tool_runtimes: list[Runtime] = []
    urls: dict[str, str] = {}
    try:
        for server in tools:
            cfg = server.config
            if not cfg.shared:
                continue
            name = _server_name(server) if isinstance(server, ServerBase) else server.name
            desc = _descriptor(server, cfg.runtime.type)
            if desc.url:
                urls[name] = desc.url
                continue
            port = _free_port()
            tool_runtime = make_runtime(cfg.runtime)
            tool_runtimes.append(tool_runtime)
            await tool_runtime.start()
            await serve_in_runtime(desc, tool_runtime, port)
            base = await tool_runtime.public_url(port) or f"http://127.0.0.1:{port}"
            urls[name] = f"{base.rstrip('/')}/mcp"
            logger.info("shared tool server '%s': %s", name, urls[name])
        yield urls
    finally:
        for tool_runtime in tool_runtimes:
            with contextlib.suppress(Exception):
                await tool_runtime.stop()


@contextlib.asynccontextmanager
async def serve_tools(
    tools: list[Tools | ServerBase],
    agent_runtime: Runtime,
    shared_urls: dict[str, str] | None = None,
):
    """Bring up a rollout's tool servers and yield `{name: url}` the harness reaches. Each
    server is placed by its `config`: a remote `url` is used as-is; a `shared` one reuses the
    eval-level instance in `shared_urls`; a `colocated` one runs in the harness's own runtime
    (reached in-sandbox, no tunnel); otherwise it runs in its OWN `runtime` (host by default),
    reached via that runtime's `public_url` or the harness runtime bridging the port — so
    different servers can run in different runtimes."""
    shared_urls = shared_urls or {}
    tool_runtimes: list[Runtime] = []  # per-rollout tool runtimes to tear down
    urls: dict[str, str] = {}
    try:
        for server in tools:
            native = isinstance(server, ServerBase)  # vf-native class vs a Tools descriptor
            name = _server_name(server) if native else server.name
            cfg = server.config
            if not native and server.url:  # already running remotely
                urls[name] = server.url
                logger.info("tool server '%s' (remote): %s", name, server.url)
            elif name in shared_urls:  # one shared instance, started eval-level
                urls[name] = shared_urls[name]
                logger.info("tool server '%s' (shared): %s", name, shared_urls[name])
            elif cfg.colocated:  # in the harness's runtime (reached in-sandbox, no tunnel)
                desc = _descriptor(server, agent_runtime.type)
                port = _free_port()
                await serve_in_runtime(desc, agent_runtime, port)
                urls[name] = f"http://127.0.0.1:{port}/mcp"
                logger.info("tool server '%s' colocated on port %d", name, port)
            else:  # its own runtime (host by default); reachability resolved per where it runs
                desc = _descriptor(server, cfg.runtime.type)
                port = _free_port()
                tool_runtime = make_runtime(cfg.runtime)
                tool_runtimes.append(tool_runtime)
                await tool_runtime.start()
                await serve_in_runtime(desc, tool_runtime, port)
                urls[name] = await _resolve_url(tool_runtime, agent_runtime, port)
                logger.info("tool server '%s' on %s: %s", name, cfg.runtime.type, urls[name])
        yield urls
    finally:
        for tool_runtime in tool_runtimes:
            with contextlib.suppress(Exception):
                await tool_runtime.stop()
