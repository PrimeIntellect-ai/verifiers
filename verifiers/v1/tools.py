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
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from pydantic import model_validator
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
      - shared: one instance for the whole eval, in its own `runtime`.
    Subclass to add the server's own knobs (the data its `@tool` methods / `respond` read).
    The server name is the class's `name` ClassVar, not a field here — it's an identity (the
    model sees `<name>_<tool>`, baked into the taskset's instruction), not a tunable knob."""

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
    url: str | None = None
    """An already-running streamable-HTTP MCP endpoint to connect to instead of launching a
    server (e.g. a public remote like DeepWiki). When set, placement is ignored — the toolset
    needs no `@tool` methods, the model just sees the remote's tools as `<name>_<tool>`."""

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
    there). Subclass to add the user's own knobs (the data its `respond` reads)."""

    colocated: bool = False
    """Run the user simulator inside the harness's runtime, reusing it (its port is published
    back to the host so the framework can still drive it). Off by default — see `ToolsetConfig`."""
    runtime: RuntimeConfig = SubprocessConfig()
    """The user simulator's own runtime, used unless `colocated` (host/subprocess by default)."""


@dataclass(frozen=True)
class _Launch:
    """Internal: how to start a tool/user server in a runtime — what `server_to_tools` produces
    and `serve_in_runtime` runs. Authors never construct this; they write a `Toolset`/`User`."""

    name: str
    script: bytes | None = None
    """A self-contained uv script (PEP 723 inline deps) that serves streamable HTTP on
    the port in `MCP_PORT` (the generated sandbox launch). Its only runtime dep is `uv`."""
    command: list[str] = field(default_factory=list)
    """Alternative to `script`: a ready-to-run argv that serves on `MCP_PORT` (the host launch,
    `python -m verifiers.v1.toolserver`, whose deps are already present)."""
    env: dict[str, str] = field(default_factory=dict)
    """Env for the script/command process (the serialized config + task refs)."""


ConfigT = TypeVar("ConfigT", bound=BaseConfig)


def _server_name(inst: ServerBase) -> str:
    """The MCP server name: the class's `name` ClassVar, else the class name snake-cased."""
    if inst.name:
        return inst.name
    return "".join(
        ("_" + c.lower() if c.isupper() else c) for c in type(inst).__name__
    ).lstrip("_")


class ServerBase(Generic[ConfigT]):
    """A vf-native server authored as a class, initialized from its config — the same shape as
    `Taskset`/`TasksetConfig`: the config (a `ToolsetConfig`/`UserConfig` subclass) is the
    serializable data (placement + the server's own knobs); the class is the behaviour. The
    framework serializes the config, launches `verifiers.v1.toolserver` in a runtime, rebuilds
    `cls(config)` there, and serves it over MCP — no FastMCP boilerplate. Subclassed by `Toolset`
    (`@tool` methods) and `User` (a `respond` hook). Declare extra PyPI deps in `deps`
    (class-level, the uv-script PEP 723 equivalent) so the framework can resolve them in any
    runtime; build expensive/non-serializable state in `setup` — set it as plain instance
    attributes (it runs in the server process)."""

    name: ClassVar[str] = ""
    """MCP server name — an identity (the model sees tools as `<name>_<tool>`), set on the
    class, not the config. Defaults to the class name snake-cased."""
    deps: ClassVar[list[str]] = []

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    async def setup(self, task) -> None:
        """Establish everything the server needs that isn't a config knob, in the server process,
        as plain instance attributes (`self.x = ...`): global state (a corpus/index/graph loaded
        from disk or a dataset), per-task input read off `task` (this rollout's task — `None` for
        a `shared` server), and initial per-rollout mutable state (counters, paths). Config knobs
        stay on `self.config`."""

    def _register(self, mcp: FastMCP) -> None:
        raise NotImplementedError


class Toolset(ServerBase[ConfigT]):
    """A tool server authored as a class: write `@vf.tool` methods (the model calls them as
    `<name>_<method>`; the docstring is the tool description), reading config off `self.config`.
    Example:

        class GlossaryToolsetConfig(vf.ToolsetConfig):
            facts: dict[str, str] = {}

        class GlossaryToolset(vf.Toolset[GlossaryToolsetConfig]):
            @vf.tool
            def lookup(self, name: str) -> str:
                return self.config.facts.get(name.lower(), "unknown")
    """

    def _register(self, mcp: FastMCP) -> None:
        for fn in discover_decorated(self, "tool"):
            mcp.add_tool(
                fn,
                name=getattr(fn, "tool_name", None) or fn.__name__,
                description=(fn.__doc__ or "").strip() or None,
            )


def serve_server(inst: ServerBase, task) -> None:
    """Run a `ServerBase` instance's MCP server (called by the `toolserver` launcher): await its
    `setup(task)`, build a FastMCP from its registered tools, and serve via `run_mcp_server`."""
    import asyncio

    from mcp.server.fastmcp import FastMCP

    asyncio.run(inst.setup(task))
    mcp = FastMCP(_server_name(inst))
    inst._register(mcp)
    run_mcp_server(mcp)


# A generated PEP 723 uv-script that rebuilds the server in any runtime: uv resolves the class's
# deps (+ verifiers + the taskset package), rebuilds `cls(config)` + the task, then serves —
# `setup(task)` establishes its global / per-task / mutable state there.
_SERVER_SCRIPT = '''\
# /// script
# requires-python = ">=3.10"
# dependencies = [{deps}]
# ///
import importlib, os

from verifiers.v1.tools import serve_server


def _ref(var):
    mod, qual = os.environ[var].split(":")
    return getattr(importlib.import_module(mod), qual)


server = _ref("VF_SERVER_REF")(_ref("VF_CONFIG_REF").model_validate_json(os.environ["VF_CONFIG"]))
task = _ref("VF_TASK_REF").model_validate_json(os.environ["VF_TASK"])
serve_server(server, task)
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


def _ref(obj: object) -> str:
    """A `"module:QualName"` import ref for an object's type — how the launcher re-imports it."""
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def server_to_tools(inst: ServerBase, runtime_type: str, task) -> _Launch:
    """Build a `_Launch` for a vf-native server. Serializes its `config` + this rollout's `task`;
    the launcher rebuilds `cls(config)` and calls `setup(task)`. On a host subprocess (deps
    already present) it runs via the `toolserver` launcher as a `command`; in any other runtime
    it generates a PEP 723 uv-script so `uv` resolves the class's declared `deps` (the deps
    answer) plus its own distribution (so the launcher can import the class + config + task) in
    that fresh sandbox."""
    cls = type(inst)
    env = {
        "VF_SERVER_REF": _ref(cls),
        "VF_CONFIG_REF": _ref(inst.config),
        "VF_CONFIG": inst.config.model_dump_json(),
        "VF_TASK_REF": _ref(task),
        "VF_TASK": task.model_dump_json(),
    }
    name = _server_name(inst)
    if runtime_type == "subprocess":
        return _Launch(
            name=name, command=[sys.executable, "-m", "verifiers.v1.toolserver"], env=env
        )
    pkg = _server_distribution(cls)
    if pkg is None:
        raise ProgramError(
            f"cannot run {cls.__qualname__} in a {runtime_type} runtime: it must live in an "
            "installed, publishable distribution so the sandbox can `uv`-install it (it resolved "
            "to no distribution). Put the class in a taskset package, or run it on a subprocess "
            "(host) runtime."
        )
    deps = ["mcp", "verifiers", *cls.deps, pkg]
    script = _SERVER_SCRIPT.format(deps=", ".join(f'"{d}"' for d in deps))
    return _Launch(name=name, script=script.encode(), env=env)


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


async def serve_in_runtime(launch: _Launch, runtime: Runtime, port: int) -> None:
    """Run `launch` inside `runtime` on `port` (background) and wait until it serves.
    A `script` runs via uv (its deps come from the script); a `command` must already be
    runnable in the runtime."""
    log = f"vf_tool_{launch.name}.log"
    if launch.script:
        script = f"vf_tool_{launch.name}.py"
        await runtime.write(script, launch.script)
        argv = ["sh", "-c", f"{_ENSURE_UV}; exec uv run --quiet {script}"]
    else:
        argv = list(launch.command)
    await runtime.run_background(argv, {**launch.env, "MCP_PORT": str(port)}, log)
    probe = await runtime.run(
        ["python3", "-c", _PROBE, f"http://127.0.0.1:{port}/mcp"], {}
    )
    if probe.exit_code != 0:
        tail = ""
        with contextlib.suppress(Exception):
            tail = (await runtime.read(log)).decode(errors="replace").strip()[-2000:]
        raise ProgramError(
            f"tool server {launch.name!r} not serving in runtime: {tail}"
        )


@contextlib.asynccontextmanager
async def serve(server: ServerBase, task, agent_runtime: Runtime | None = None, for_host: bool = False):
    """The single internal launcher for a vf-native server — a `Toolset` OR a `User`. Brings it
    up in its configured placement and yields one reachable URL, tearing down any runtime it
    owns. Placement comes from `server.config`:
      - `colocated` (with an `agent_runtime`): runs in the harness's own runtime, reusing it;
      - otherwise: its OWN `runtime` (host by default), started and stopped here.
    (A remote `url` toolset is short-circuited by the caller — it isn't launched.) Reachability
    depends on who consumes it: `for_host` (a user sim the framework drives) yields a
    host-reachable URL; otherwise (a tool the model calls) a harness-reachable one — localhost
    in-sandbox when colocated, else the tool runtime's `public_url` bridged by the harness's
    `expose` (or, eval-level with no `agent_runtime`, `public_url` or localhost for a shared one)."""
    cfg = server.config
    own = None
    if cfg.colocated and agent_runtime is not None:
        runtime = agent_runtime
    else:
        own = make_runtime(cfg.runtime)
        await own.start()
        runtime = own
    try:
        port = _free_port()
        await serve_in_runtime(server_to_tools(server, runtime.type, task), runtime, port)
        local = f"http://127.0.0.1:{port}"
        if for_host:  # the framework reaches it from the host
            base = await runtime.public_url(port) or local
        elif runtime is agent_runtime:  # colocated tool: the model reaches it in-sandbox
            base = local
        elif agent_runtime is not None:  # own-runtime tool: the harness bridges to it
            base = await runtime.public_url(port) or await agent_runtime.expose(port)
        else:  # shared tool, eval-level (no single agent to bridge through)
            base = await runtime.public_url(port) or local
        yield f"{base.rstrip('/')}/mcp"
    finally:
        if own is not None:
            with contextlib.suppress(Exception):
                await own.stop()


@contextlib.asynccontextmanager
async def serve_shared(toolsets: list[Toolset], task):
    """Start the SHARED tool servers (placement `shared`) ONCE for a whole eval, each in its OWN
    `runtime`, and yield `{name: url}` reachable by every rollout's harness (a prime tool runtime
    publishes its port; a host one is localhost, for host-network harnesses). Torn down when the
    eval ends. Used by `Environment` so an expensive corpus is built once, not per rollout."""
    urls: dict[str, str] = {}
    async with contextlib.AsyncExitStack() as stack:
        for toolset in toolsets:
            cfg = toolset.config
            if not cfg.shared:
                continue
            name = _server_name(toolset)
            if cfg.url:  # already running remotely
                urls[name] = cfg.url
            else:
                urls[name] = await stack.enter_async_context(serve(toolset, task))
            logger.info("shared tool server '%s': %s", name, urls[name])
        yield urls


@contextlib.asynccontextmanager
async def serve_tools(
    toolsets: list[Toolset],
    agent_runtime: Runtime,
    task,
    shared_urls: dict[str, str] | None = None,
):
    """Bring up a rollout's tool servers and yield `{name: url}` the harness reaches. A `shared`
    toolset reuses the eval-level instance in `shared_urls`; the rest are launched by `serve`
    (placement off each one's `config`, the rollout's `task` for its `setup`) — so different
    servers can run in different runtimes."""
    shared_urls = shared_urls or {}
    urls: dict[str, str] = {}
    async with contextlib.AsyncExitStack() as stack:
        for toolset in toolsets:
            name = _server_name(toolset)
            cfg = toolset.config
            if cfg.url:  # already running remotely
                urls[name] = cfg.url
                logger.info("tool server '%s' (remote): %s", name, cfg.url)
            elif name in shared_urls:  # one shared instance, started eval-level
                urls[name] = shared_urls[name]
                logger.info("tool server '%s' (shared): %s", name, shared_urls[name])
            else:
                urls[name] = await stack.enter_async_context(serve(toolset, task, agent_runtime))
                logger.info("tool server '%s': %s", name, urls[name])
        yield urls
