"""Tools: how a task gives the harness tools, via `vf.Toolset`s it declares from `Taskset.tools`.

A `Toolset` (and a `User`) is authored as a vf-native class. The framework launches it with a
generic entrypoint (`python -m verifiers.v1.toolserver`) that imports the real class from its
(installed) env module and serves it over streamable HTTP on `MCP_PORT`. On a host (`subprocess`)
runtime that's the eval's own interpreter — `verifiers` and the env module are already installed,
nothing is fetched. In a sandbox the env module is uploaded and `uv pip install`ed (pulling
`verifiers`, a git-pinned dependency of the env package, plus the env's own deps), then the same
entrypoint runs. A `Toolset` can instead point at an already-running remote endpoint
(`ToolsetConfig.url`, e.g. deepwiki).

`serve` is the single internal launcher (any server, any placement). `serve_tools` brings a
task's servers up for a rollout (each in its `config`'s placement — colocated, own runtime);
`serve_shared` brings up shared ones once per eval. `serve_server`/`run_mcp_server` run the
server-side serve loop (called by the entrypoint). The wire types the model sees (`Tool`,
`ToolCall`, …) live in `types`.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import random
import shlex
import socket
import sys
import tarfile
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

# `verifiers` isn't on PyPI at the v1 version, so a sandbox installs it from git (the env package
# declares it as a dependency; this supplies the pin). Locally the editable workspace is used.
_VERIFIERS_PIN = "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@be2bd96"


class ToolsetConfig(BaseConfig):
    """Where one tool server runs (placement). The default — its own host (`subprocess`)
    runtime — is the cheap one: the server runs in the eval process's environment, where
    `verifiers` and the env module are already installed (nothing to fetch), and the harness
    reaches it over the host network (docker `--network host`) or a tunnel (prime). `colocated`
    and `shared` trade that off:
      - neither (default): its own `runtime`, per rollout (host by default).
      - colocated: in the harness's OWN runtime, per rollout (no tunnel). In a sandbox this means
        the env module is uploaded and `uv pip install`ed there (pulling git-pinned `verifiers`),
        so it costs a per-rollout install.
      - shared: one instance for the whole eval, in its own `runtime`.
    Subclass to add the server's own knobs (the data its `@tool` methods / `respond` read).
    The server name is the class's `name` ClassVar, not a field here — it's an identity (the
    model sees `<name>_<tool>`, baked into the taskset's instruction), not a tunable knob."""

    colocated: bool = False
    """Run the server inside the harness's runtime (reached in-sandbox, no tunnel). Off by
    default — on the host it's free, but in a sandbox the harness runtime must install the env
    package + `verifiers` (a per-rollout cost), so prefer the default own-runtime placement
    unless co-locating genuinely helps."""
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
    """Internal: everything `serve_in_runtime` needs to start a server with the generic entrypoint
    (`python -m verifiers.v1.toolserver`). `env` carries the serialized class/config/task refs the
    entrypoint reads; `source_dir` is the env package's local directory (the one with `pyproject`),
    uploaded + installed when the runtime is a sandbox (`None` for a non-local/ambient module).
    What `server_to_launch` produces; authors never construct it."""

    name: str
    env: dict[str, str] = field(default_factory=dict)
    source_dir: str | None = None


ConfigT = TypeVar("ConfigT", bound=BaseConfig)


class ServerBase(Generic[ConfigT]):
    """A vf-native server authored as a class, initialized from its config — the same shape as
    `Taskset`/`TasksetConfig`: the config (a `ToolsetConfig`/`UserConfig` subclass) is the
    serializable data (placement + the server's own knobs); the class is the behaviour. The
    framework launches it with the generic entrypoint (`python -m verifiers.v1.toolserver`), which
    imports this class from its env module, rebuilds `cls(config)`, and serves over MCP — no
    FastMCP boilerplate. Subclassed by `Toolset` (`@tool` methods) and `User` (a `respond` hook).
    Build expensive/non-serializable state in `setup` — set it as plain instance attributes (it
    runs in the server process). The server's deps come from its env package's `pyproject` (the
    install in a sandbox), so the class may freely `import verifiers`, import siblings, and use
    module globals; `deps` is an extra hook for ad-hoc PyPI requirements."""

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


def _source_dir(cls: type) -> str | None:
    """The local directory of `cls`'s env package — the nearest ancestor of its module file that
    holds a `pyproject.toml` (what gets uploaded + installed in a sandbox). `None` when the module
    has no on-disk package (a hub install in site-packages, or a built-in)."""
    module = sys.modules.get(cls.__module__)
    path = getattr(module, "__file__", None)
    if not path:
        return None
    for parent in Path(path).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return str(parent)
    return None


def server_to_launch(inst: ServerBase, task) -> _Launch:
    """Build the `_Launch` for a vf-native server: the env the generic entrypoint
    (`verifiers.v1.toolserver`) reads to rebuild it — the class and config-class refs
    (`module:qualname`), and the `config` + this rollout's `task` as JSON — plus the env package's
    local `source_dir` (uploaded + installed when the runtime is a sandbox)."""
    cls, cfg_cls, task_cls = type(inst), type(inst.config), type(task)
    env = {
        "VF_SERVER": f"{cls.__module__}:{cls.__qualname__}",
        "VF_CONFIG_CLS": f"{cfg_cls.__module__}:{cfg_cls.__qualname__}",
        "VF_TASK_CLS": f"{task_cls.__module__}:{task_cls.__qualname__}",
        "VF_CONFIG": inst.config.model_dump_json(),
        "VF_TASK": task.model_dump_json(),
    }
    return _Launch(name=inst.server_name, env=env, source_dir=_source_dir(cls))


def run_mcp_server(mcp: "FastMCP") -> None:
    """Serve a FastMCP server on the port the harness passes via `MCP_PORT`, mounting
    streamable HTTP at `/mcp`. Called by the server entrypoint (`verifiers.v1.toolserver`)."""
    import uvicorn

    port = int(os.environ["MCP_PORT"])
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    config = uvicorn.Config(
        mcp.streamable_http_app(), host=host, port=port, log_level="critical"
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


async def _install_in_sandbox(launch: _Launch, runtime: Runtime) -> str:
    """Make the env module importable in a sandbox: upload its package directory (a tarball over
    `write`), create a venv, and `uv pip install` git-pinned `verifiers` then the package (which
    pulls the env's own deps). Returns the venv's python. The env package's `verifiers` dependency
    is satisfied by the pre-installed git pin, so it resolves to the v1 version, not PyPI."""
    if launch.source_dir is None:
        raise ProgramError(
            f"server {launch.name!r} runs in a {runtime.type} runtime but its module is not a "
            "local package (no pyproject) — sandbox launch needs a local env package to upload"
        )
    src = Path(launch.source_dir)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(src, arcname=src.name)
    await runtime.write("/tmp/vf-env/pkg.tar.gz", buf.getvalue())
    venv, pkg = "/tmp/vf-venv", f"/tmp/vf-env/{src.name}"
    setup = (
        f"{_ENSURE_UV}; set -e; "
        # the git-pinned verifiers install needs a git client (slim base images lack one)
        "command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git >/dev/null; }; "
        "tar -xzf /tmp/vf-env/pkg.tar.gz -C /tmp/vf-env && "
        f"uv venv {venv} && "
        f"uv pip install --python {venv} {shlex.quote(_VERIFIERS_PIN)} && "
        f"uv pip install --python {venv} {shlex.quote(pkg)}"
    )
    result = await runtime.run(["sh", "-c", setup], {})
    if result.exit_code != 0:
        raise ProgramError(
            f"server {launch.name!r} install failed in runtime: "
            f"{(result.stderr or result.stdout).strip()[-2000:]}"
        )
    return f"{venv}/bin/python"


async def serve_in_runtime(launch: _Launch, runtime: Runtime, port: int) -> None:
    """Start the server inside `runtime` on `port` (background, via the generic entrypoint) and
    wait until it serves. On a host (`subprocess`) runtime it runs with the eval's own interpreter
    — `verifiers` and the env module are already installed, nothing is fetched. In a sandbox the
    env package is uploaded + installed first (`_install_in_sandbox`), then run with that venv."""
    log = f"vf_tool_{launch.name}.log"
    if runtime.type == "subprocess":  # host: verifiers + env module already installed
        python = sys.executable
    else:  # sandbox: upload + install the env package (pulls git-pinned verifiers)
        python = await _install_in_sandbox(launch, runtime)
    argv = [python, "-m", "verifiers.v1.toolserver"]
    env = {**launch.env, "MCP_PORT": str(port)}
    if runtime.published_port is not None:  # a self-publishing runtime (modal) forwards to all
        env["MCP_HOST"] = "0.0.0.0"  # interfaces, not just loopback
    await runtime.run_background(argv, env, log)
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
        port = runtime.published_port or _free_port()
        await serve_in_runtime(server_to_launch(server, task), runtime, port)
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
