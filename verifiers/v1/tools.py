"""Tools: how a task gives the harness tools, via `vf.Toolset`s it declares from `Taskset.tools`.

A `Toolset` (and a `User`) is authored as a vf-native class; the framework renders it to a plain
PEP 723 uv-script (`server_to_tools` → `_render_script`) that, launched in a runtime, rebuilds
the class and serves it over streamable HTTP on `MCP_PORT`. On a host (`subprocess`) runtime the
script is run with the eval's own interpreter — deps already installed, nothing fetched, no
publishing; in a sandbox it's `uv run`, resolving the header's deps from PyPI. A `Toolset` can
instead point at an already-running remote endpoint (`ToolsetConfig.url`, e.g. deepwiki).

`serve` is the single internal launcher (any server, any placement). `serve_tools` brings a
task's servers up for a rollout (each in its `config`'s placement — colocated, own runtime);
`serve_shared` brings up shared ones once per eval. `run_mcp_server` is the server-side serve
loop. The wire types the model sees (`Tool`, `ToolCall`, …) live in `types`.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import logging
import os
import random
import socket
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass(frozen=True)
class _Launch:
    """Internal: a rendered PEP 723 uv-script that serves a tool/user server over streamable HTTP
    on `MCP_PORT`, plus the env (serialized config + task refs) it reads. What `server_to_tools`
    produces and `serve_in_runtime` `uv run`s. Authors never construct this — they write a
    `Toolset`/`User` and the framework renders it."""

    name: str
    script: bytes
    env: dict[str, str] = field(default_factory=dict)


ConfigT = TypeVar("ConfigT", bound=BaseConfig)


class ServerBase(Generic[ConfigT]):
    """A vf-native server authored as a class, initialized from its config — the same shape as
    `Taskset`/`TasksetConfig`: the config (a `ToolsetConfig`/`UserConfig` subclass) is the
    serializable data (placement + the server's own knobs); the class is the behaviour. The
    framework renders a PEP 723 uv-script from it (`server_to_tools`), `uv run`s it in a runtime
    where it rebuilds `cls(config)` and serves over MCP — no FastMCP boilerplate. Subclassed by `Toolset`
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

    @property
    def server_name(self) -> str:
        """The MCP server name: the class's `name` ClassVar, else the class name snake-cased."""
        return self.name or "".join(
            ("_" + c.lower() if c.isupper() else c) for c in type(self).__name__
        ).lstrip("_")

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
    """Run a `ServerBase` instance's MCP server (called by the rendered server script): await its
    `setup(task)`, build a FastMCP from its registered tools, and serve via `run_mcp_server`."""
    import asyncio

    from mcp.server.fastmcp import FastMCP

    asyncio.run(inst.setup(task))
    mcp = FastMCP(inst.server_name)
    inst._register(mcp)
    run_mcp_server(mcp)


# The server-side runtime, vendored verbatim into every rendered script (NOT imported at serve
# time) — so the script depends on neither `verifiers` nor the taskset package, only public PyPI.
_SERVERKIT = (Path(__file__).parent / "_serverkit.py").read_text()


def _render_script(inst: ServerBase, deps: list[str]) -> bytes:
    """Render a STANDALONE PEP 723 uv-script for `inst`: a public-PyPI dep header
    (`mcp`/`pydantic`/`uvicorn` + the class's own `deps`), the vendored runtime, then the server's
    config + class source inlined and reconstructed from the env. It imports neither `verifiers`
    nor the taskset package — so it runs in any runtime (and a fresh sandbox) with no publishing."""
    cls = type(inst)
    cfg_cls = type(inst.config)
    # An author config subclass carries the server's data and is inlined; the bare runtime base
    # (placement only — irrelevant here) is already in the vendored kit.
    cfg_src = "" if cfg_cls.__module__.startswith("verifiers") else inspect.getsource(cfg_cls)
    header = "\n".join(
        ["# /// script", '# requires-python = ">=3.10"', "# dependencies = ["]
        + [f'#   "{d}",' for d in ["mcp", "pydantic", "uvicorn", *deps]]
        + ["# ]", "# ///"]
    )
    body = f'''
import os as _os, sys as _sys

vf = _sys.modules[__name__]  # the authored class refers to `vf.Toolset` / `@vf.tool` / `vf.UserConfig`

{cfg_src}

{inspect.getsource(cls)}

_inst = {cls.__name__}({cfg_cls.__name__}.model_validate_json(_os.environ["VF_CONFIG"]))
serve_server(_inst, Task.model_validate_json(_os.environ["VF_TASK"]))
'''
    return (header + "\n" + _SERVERKIT + body).encode()


def server_to_tools(inst: ServerBase, task) -> _Launch:
    """Render a vf-native server to a `_Launch` — a standalone PEP 723 uv-script (the vendored
    runtime + the server's inlined config/class) that rebuilds `cls(config)`, runs `setup(task)`,
    and serves. The `config` + this rollout's `task` cross the wire as JSON in the env; the
    script's only deps are `mcp`/`pydantic`/`uvicorn` + the class's declared `deps` — all public
    PyPI, no `verifiers` or taskset package, so it needs nothing pre-installed in a sandbox."""
    env = {
        "VF_CONFIG": inst.config.model_dump_json(),
        "VF_TASK": task.model_dump_json(),
    }
    return _Launch(
        name=inst.server_name, script=_render_script(inst, list(type(inst).deps)), env=env
    )


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
    """Start the launch's script inside `runtime` on `port` (background) and wait until it serves.
    On a host (`subprocess`) runtime it runs with the eval's own interpreter — deps already
    installed, so the PEP 723 header is ignored and nothing is fetched; in any other runtime it's
    `uv run`, resolving the header's deps from PyPI. Written to a stable, content-addressed path
    so uv keys one resolved environment per distinct script, shared across rollouts."""
    log = f"vf_tool_{launch.name}.log"
    path = f"/tmp/vf-scripts/{hashlib.sha256(launch.script).hexdigest()}.py"
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"  # publish atomically — concurrent rollouts share it
    await runtime.write(tmp, launch.script)
    await runtime.run(["sh", "-c", f"mkdir -p /tmp/vf-scripts && mv -f {tmp} {path}"], {})
    if runtime.type == "subprocess":  # host: deps already installed — run with the eval's python
        argv = [sys.executable, path]
    else:  # sandbox: uv resolves the script's PEP 723 deps from PyPI
        argv = ["sh", "-c", f"{_ENSURE_UV}; exec uv run --quiet {path}"]
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
        await serve_in_runtime(server_to_tools(server, task), runtime, port)
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
            name = toolset.server_name
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
            name = toolset.server_name
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
