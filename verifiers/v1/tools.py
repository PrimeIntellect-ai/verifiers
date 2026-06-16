"""Tools: how a task gives the harness tools, via `vf.Toolset`s it declares from `Taskset.tools`.

A `Toolset` (and a `User`) is authored as a vf-native class whose env module ends with
`if __name__ == "__main__": <Server>.run()`. The framework launches it by running that module
(`python -m <module>`); `ServerBase.run()` rebuilds the server from the environment and serves it
over streamable HTTP on `MCP_PORT`. On a host (`subprocess`) runtime that's the eval's own
interpreter — `verifiers` and the env module are already installed, nothing is fetched. In a sandbox
the working-tree `verifiers` source and the env module are uploaded and `uv pip install`ed (deps
resolve from PyPI), then the module runs the same way — no publish or pin to keep in sync. A
`Toolset` can instead point at an already-running remote endpoint (`ToolsetConfig.url`, e.g. deepwiki).

`serve` is the single internal launcher (any server, any placement). `serve_tools` brings a
task's servers up for a rollout (each in its `config`'s placement — colocated, own runtime);
`serve_shared` brings up shared ones once per eval. `ServerBase.run` → `ServerBase._serve` is the
server-side serve loop (in the launched process). The wire types the model sees (`Tool`,
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
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, get_args

from pydantic import model_validator
from pydantic_config import BaseConfig

from verifiers.v1.decorators import discover_decorated
from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
    host_endpoint,
    make_runtime,
)
from verifiers.v1.runtimes.base import _ENSURE_UV, SERVICE_PORT

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# The verifiers source tree's wheel-build inputs — uploaded into a sandbox so it installs the
# developer's working-tree verifiers (deps resolve from PyPI off the uploaded pyproject), with no
# publish or git pin to keep in sync.
VF_BUILD_INPUTS = ["pyproject.toml", "README.md", "LICENSE", "verifiers"]


class ToolsetConfig(BaseConfig):
    """Where one tool server runs (placement). The default — its own host (`subprocess`)
    runtime — is the cheap one: the server runs in the eval process's environment, where
    `verifiers` and the env module are already installed (nothing to fetch), and the harness
    reaches it over the host network (docker `--network host`) or a tunnel (prime). `colocated`
    and `shared` trade that off:
      - neither (default): its own `runtime`, per rollout (host by default).
      - colocated: in the harness's OWN runtime, per rollout (no tunnel). In a sandbox this means
        the `verifiers` source + the env module are uploaded and `uv pip install`ed there, so it
        costs a per-rollout install.
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
    def reject_colocated_and_shared(self) -> "ToolsetConfig":
        if self.colocated and self.shared:
            raise ValueError("colocated and shared are mutually exclusive")
        return self


ConfigT = TypeVar("ConfigT", bound=BaseConfig)


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
        raise TypeError(f"{cls.__name__} must parameterize its config, e.g. Toolset[MyConfig]")

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
            task = _import_ref(os.environ["VF_TASK_CLS"]).model_validate_json(
                os.environ["VF_TASK"]
            )
        cls(config)._serve(task)


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


def _import_ref(ref: str) -> object:
    """Resolve a `module:qualname` reference (e.g. `glossary_v1:GlossaryTask`) to the object."""
    import importlib

    module_name, _, qualname = ref.partition(":")
    obj: object = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


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


def _tar_source(src: Path, members: list[str] | None = None) -> bytes:
    """Gzipped tarball of a local package dir, rooted at `src.name/` and excluding `__pycache__`.
    `members` limits it to those top-level entries (the verifiers tree only needs its package +
    project files); otherwise the whole dir (a small env package)."""

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return None if "__pycache__" in info.name.split("/") else info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for member in members or [""]:
            path = src / member if member else src
            if path.exists():
                tar.add(path, arcname=f"{src.name}/{member}".rstrip("/"), filter=keep)
    return buf.getvalue()


def _verifiers_root() -> Path:
    """The verifiers source checkout — the dir holding its `pyproject.toml`, above the package."""
    import verifiers

    root = Path(verifiers.__file__).resolve().parent.parent
    if not (root / "pyproject.toml").exists():
        raise ProgramError(
            "verifiers is not a source checkout (no pyproject above the package), so it can't be "
            "uploaded to a sandbox; run sandboxed servers from a verifiers source install"
        )
    return root


async def _install_in_sandbox(server: ServerBase, runtime: Runtime) -> str:
    """Make `server`'s env module importable in a sandbox: upload the working-tree `verifiers`
    source and the env package (tarballs over `write`), create a venv, and `uv pip install` both —
    verifiers first (deps resolve from PyPI off its pyproject), then the env package (its `verifiers`
    dep already satisfied). Returns the venv's python. Uses the developer's current code — no publish
    or pin to keep in sync."""
    source_dir = _source_dir(type(server))
    if source_dir is None:
        raise ProgramError(
            f"server {server.server_name!r} runs in a {runtime.type} runtime but its module is not "
            "a local package (no pyproject) — sandbox launch needs a local env package to upload"
        )
    root = "/tmp/vf-src"
    vf, env = _verifiers_root(), Path(source_dir)
    await runtime.write(f"{root}/{vf.name}.tar.gz", _tar_source(vf, VF_BUILD_INPUTS))
    await runtime.write(f"{root}/{env.name}.tar.gz", _tar_source(env))
    venv = "/tmp/vf-venv"
    setup = (
        f"{_ENSURE_UV}; set -e; "
        f'for t in {root}/*.tar.gz; do tar -xzf "$t" -C {root}; done && '
        f"uv venv {venv} && "
        f"uv pip install --python {venv} {root}/{shlex.quote(vf.name)} && "
        f"uv pip install --python {venv} {root}/{shlex.quote(env.name)}"
    )
    result = await runtime.run(["sh", "-c", setup], {})
    if result.exit_code != 0:
        raise ProgramError(
            f"server {server.server_name!r} install failed in runtime: "
            f"{(result.stderr or result.stdout).strip()[-2000:]}"
        )
    return f"{venv}/bin/python"


async def serve_in_runtime(
    server: ServerBase, task, runtime: Runtime, port: int
) -> None:
    """Start `server` inside `runtime` on `port` (background, by running its env module —
    `python -m <module>`, whose `__main__` calls `ServerBase.run()`) and wait until it serves. The
    `config` + this rollout's `task` cross to the server as env JSON (a `shared` server passes
    `None`). On a host (`subprocess`) runtime it runs with the eval's own interpreter — `verifiers`
    and the env module are already installed, nothing is fetched. In a sandbox the working-tree
    `verifiers` source + the env package are uploaded and installed first (`_install_in_sandbox`),
    then run from that venv."""
    env = {"VF_CONFIG": server.config.model_dump_json(), "MCP_PORT": str(port)}
    if task is not None:
        env["VF_TASK_CLS"] = f"{type(task).__module__}:{type(task).__qualname__}"
        env["VF_TASK"] = task.model_dump_json()
    if runtime.published_port is not None:  # a self-publishing runtime (modal/prime) forwards to
        env["MCP_HOST"] = "0.0.0.0"  # all interfaces, not just loopback
    if runtime.type == "subprocess":  # host: verifiers + env module already installed
        python = sys.executable
    else:  # sandbox: upload + install the verifiers source + the env package
        python = await _install_in_sandbox(server, runtime)
    log = f"vf_tool_{server.server_name}.log"
    await runtime.run_background([python, "-m", type(server).__module__], env, log)
    probe = await runtime.run(
        ["python3", "-c", _PROBE, f"http://127.0.0.1:{port}/mcp"], {}
    )
    if probe.exit_code != 0:
        tail = ""
        with contextlib.suppress(Exception):
            tail = (await runtime.read(log)).decode(errors="replace").strip()[-2000:]
        raise ProgramError(
            f"tool server {server.server_name!r} not serving in runtime: {tail}"
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
    in-sandbox when colocated, else the tool runtime's `expose` (its published URL) or, for a
    host-side tool reached by an in-sandbox harness, a `host_endpoint` tunnel to the host port."""
    cfg = server.config
    shared = getattr(cfg, "shared", False)
    if shared and task is not None:
        raise ValueError(
            f"shared server {server.server_name!r} was launched with a task, but a `shared` server "
            "is built once for the whole eval and must be task-agnostic — it receives no task. "
            "Drop `shared` to run it per-rollout (with the task), or make its `setup` task-independent."
        )
    if shared and type(server).setup_task is not ServerBase.setup_task:
        logger.warning(
            "shared server %r overrides `setup_task`, but `setup_task` is NEVER called for a shared "
            "server (it's built once, task-agnostic) — its per-task logic will not run. Move "
            "task-agnostic work into `setup`, or drop `shared` to run it per-rollout.",
            server.server_name,
        )
    async with contextlib.AsyncExitStack() as stack:
        if cfg.colocated and agent_runtime is not None:
            runtime = agent_runtime
        else:
            runtime = make_runtime(cfg.runtime)
            await runtime.start()
            stack.push_async_callback(runtime.stop)
        port = runtime.published_port or _free_port()
        await serve_in_runtime(server, task, runtime, port)
        local = f"http://127.0.0.1:{port}"
        if for_host:  # the framework reaches it from the host
            base = await runtime.expose(port) or local
        elif runtime is agent_runtime:  # colocated tool: the model reaches it in-sandbox
            base = local
        elif agent_runtime is not None:  # own-runtime tool: the harness bridges to it
            base = await runtime.expose(port) or await stack.enter_async_context(
                host_endpoint(port, agent_runtime.is_local)
            )
        else:  # shared tool, eval-level (no single agent to bridge through)
            base = await runtime.expose(port) or local
        yield f"{base.rstrip('/')}/mcp"


@contextlib.asynccontextmanager
async def serve_shared(toolsets: list[Toolset]):
    """Start the SHARED tool servers (placement `shared`) ONCE for a whole eval, each in its OWN
    `runtime`, and yield `{name: url}` reachable by every rollout's harness (a prime tool runtime
    publishes its port; a host one is localhost, for host-network harnesses). Torn down when the
    eval ends. Used by `Environment` so an expensive corpus is built once, not per rollout. A shared
    server is task-agnostic, so its `setup` gets no task (`serve(toolset, None)`)."""
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
                urls[name] = await stack.enter_async_context(serve(toolset, None))
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
