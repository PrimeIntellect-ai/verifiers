from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import secrets
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import PathDistribution, distribution, packages_distributions
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from verifiers.v1.errors import ToolsetError
from verifiers.v1.mcp.server import (
    STATE_ROUTE_PARAM,
    STATE_SIGNATURE_PARAM,
    STATE_URL_PARAM,
    ServerBase,
    state_signature,
)
from verifiers.v1.mcp.toolset import ToolsetConfig, ToolsetTimeoutConfig
from verifiers.v1.runtimes import (
    PrimeBackend,
    PrimeConfig,
    Runtime,
    provision_backend,
)
from verifiers.v1.runtimes.base import _ENSURE_UV
from verifiers.v1.runtimes.private_network import eval_lan, join_lan, lan_host
from verifiers.v1.state import State
from verifiers.v1.utils.aio import run_shielded

if TYPE_CHECKING:
    from verifiers.v1.mcp.toolset import Toolset
    from verifiers.v1.runtimes.deployment import Deployment
    from verifiers.v1.session import MCPProxy

logger = logging.getLogger(__name__)

_PACKAGE_BUILD_TIMEOUT_SECONDS = 300


@dataclass
class _PackageBuildState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    build: asyncio.Task[tuple[str, bytes]] | None = None
    users: int = 0


# Coordination and caching are process-local. Separate worker processes each
# build once; within an event loop, concurrent rollouts share the completed artifact.
_PACKAGE_BUILD_STATES: dict[
    tuple[Path, asyncio.AbstractEventLoop], _PackageBuildState
] = {}

# Any HTTP response, including MCP's 406 to a bare GET, proves the server is listening.
_PROBE = """
import sys, time, urllib.error, urllib.request
deadline = time.monotonic() + float(sys.argv[2])
while True:
    try:
        urllib.request.urlopen(sys.argv[1], timeout=2); sys.exit(0)
    except urllib.error.HTTPError:
        sys.exit(0)
    except Exception:
        pass
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        sys.exit(1)
    time.sleep(min(0.1, remaining))
"""


@cache
def _package_dir(cls: type) -> str:
    module = sys.modules.get(cls.__module__)
    path = getattr(module, "__file__", None)
    if path:
        path = Path(path).resolve()
        # An installed wheel owns its recorded files, even when its venv lives
        # inside an unrelated source checkout with a pyproject.toml above it.
        for name in packages_distributions().get(cls.__module__.split(".")[0], []):
            files = distribution(name).files or []
            if any(Path(file.locate()).resolve() == path for file in files):
                for file in files:
                    if file.name == "WHEEL" and file.parent.suffix == ".dist-info":
                        return str(Path(file.locate()).parent)
        for parent in path.parents:
            if parent.name in {"site-packages", "dist-packages"}:
                break
            if (parent / "pyproject.toml").exists():
                return str(parent)
    raise ToolsetError(
        f"cannot package {cls.__module__!r} for a sandbox: "
        "no installed wheel or source project found"
    )


@cache
def _build_package(src: Path) -> tuple[str, bytes]:
    """Transfer installed wheel files or build a source project's PEP 517 sdist."""
    with tempfile.TemporaryDirectory(prefix="vf-package-") as directory:
        out = Path(directory)
        if src.suffix == ".dist-info":
            dist = PathDistribution(src)
            files = dist.files
            if not files:
                raise ToolsetError(f"cannot package {src}: missing RECORD")
            package = out / "package"
            scripts = {
                entry.name
                for entry in dist.entry_points
                if entry.group in {"console_scripts", "gui_scripts"}
            }
            for file in files:
                # Installers regenerate entry-point scripts for the target Python.
                if ".." in file.parts or file.is_absolute():
                    if (
                        file.name in scripts
                        and Path(file.locate()).parent.resolve()
                        == Path(sysconfig.get_path("scripts")).resolve()
                    ):
                        continue
                    raise ToolsetError(
                        f"cannot package {src}: installed file outside site-packages: {file}"
                    )
                if file.suffix == ".pyc" or (
                    file.parent.name == src.name
                    and file.name
                    in {
                        "RECORD",
                        "RECORD.jws",
                        "RECORD.p7s",
                        "INSTALLER",
                        "REQUESTED",
                        "direct_url.json",
                        "uv_cache.json",
                    }
                ):
                    continue
                target = package / file
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file.locate(), target)
            command = [
                "uv",
                "tool",
                "run",
                "--from",
                "wheel==0.48.0",
                "wheel",
                "pack",
                "--dest-dir",
                str(out),
                str(package),
            ]
        else:
            command = [
                "uv",
                "build",
                "--sdist",
                "--no-create-gitignore",
                "--color",
                "never",
                "--out-dir",
                str(out),
                ".",
            ]
        try:
            # Run from the source root so uv discovers this project's workspace,
            # uv.toml, indexes, constraints, and Python configuration rather than
            # inheriting configuration from the launcher's working directory.
            result = subprocess.run(
                command,
                cwd=src,
                capture_output=True,
                text=True,
                check=False,
                timeout=_PACKAGE_BUILD_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as e:
            raise ToolsetError(
                f"cannot build package for {src}: uv is not installed"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise ToolsetError(
                "package build timed out after "
                f"{_PACKAGE_BUILD_TIMEOUT_SECONDS}s for {src}"
            ) from e
        if result.returncode != 0:
            detail = "\n".join(
                part.strip() for part in (result.stdout, result.stderr) if part.strip()
            )
            raise ToolsetError(f"package build failed for {src}: {detail[-2000:]}")
        artifacts = [path for path in out.iterdir() if path.is_file()]
        if len(artifacts) != 1:
            names = ", ".join(sorted(path.name for path in artifacts)) or "none"
            raise ToolsetError(
                f"package build for {src} produced {len(artifacts)} distributions: {names}"
            )
        artifact = artifacts[0]
        return artifact.name, artifact.read_bytes()


async def _cached_package(src: Path) -> tuple[str, bytes]:
    """Build once per source without parking waiters in the thread pool."""
    src = src.resolve()
    key = (src, asyncio.get_running_loop())
    state = _PACKAGE_BUILD_STATES.get(key)
    if state is None:
        state = _PackageBuildState()
        _PACKAGE_BUILD_STATES[key] = state
    state.users += 1
    try:
        async with state.lock:
            # Coordinate before entering the default executor so concurrent
            # same-source waiters do not occupy executor threads.
            if state.build is None:
                state.build = asyncio.create_task(
                    asyncio.to_thread(_build_package, src)
                )
            build = state.build
            try:
                return await asyncio.shield(build)
            except asyncio.CancelledError:
                # Cancelling to_thread does not stop its worker. Keep the lock until
                # that worker exits so a replacement caller cannot overlap the build.
                with contextlib.suppress(Exception):
                    await build
                raise
    finally:
        state.users -= 1
        if state.users == 0 and _PACKAGE_BUILD_STATES.get(key) is state:
            del _PACKAGE_BUILD_STATES[key]


async def _install_in_sandbox(server: ServerBase, target: Runtime) -> str:
    # Install as root; a service's named user only runs the server.
    runtime = target if target.user is None else target.with_user(None)
    # Prime VMs mount /tmp as a small tmpfs, while the runtime workdir lives on
    # the VM's root disk. Keep source, build scratch space, and uv's cache on the
    # durable runtime filesystem so ordinary dependency installs cannot exhaust
    # the tmpfs.
    # Every runtime runs commands in its working directory; not every config names it.
    located = await runtime.run(["pwd"], {})
    workdir = located.stdout.strip()
    if located.exit_code or not workdir:
        raise ToolsetError(
            f"server {server.server_name!r}: cannot locate the runtime's working "
            f"directory: {located.stderr.strip()}"
        )
    root = str(PurePosixPath(workdir) / ".vf-src")
    temp = str(PurePosixPath(workdir) / ".vf-tmp")
    cache = str(PurePosixPath(workdir) / ".vf-uv-cache")
    venv = str(PurePosixPath(workdir) / ".vf-venv")
    python_dir = str(PurePosixPath(workdir) / ".vf-python")
    root_q, temp_q, cache_q, venv_q = map(shlex.quote, (root, temp, cache, venv))
    # Servers placed in one service install into one physical environment.
    # Serialize its mutations and only remember sources after a successful install.
    async with runtime.backend.cache.mcp_lock:
        sources = dict.fromkeys((_package_dir(ServerBase), _package_dir(type(server))))
        pending = [
            source
            for source in sources
            if source not in runtime.backend.cache.mcp_sources
        ]
        if pending:
            setup = (
                f"set -e; mkdir -p {root_q} {temp_q} {cache_q}; "
                f"export TMPDIR={temp_q} UV_CACHE_DIR={cache_q}; "
                'export PATH="$HOME/.local/bin:$PATH"; '
            )
            if target.user is not None:
                setup += f"export UV_PYTHON_INSTALL_DIR={shlex.quote(python_dir)}; "
            if not runtime.backend.cache.mcp_sources:
                # Failed installs can leave the venv behind; retain it when retrying.
                setup += f"{_ENSURE_UV}; uv venv --allow-existing {venv_q}; "
            # Drain remote writes and installs before cancellation releases the lock.
            for source in pending:
                name, data = await _cached_package(Path(source))
                remote = f"{root}/{name}"
                await run_shielded(runtime.write(remote, data))
                setup += f"uv pip install --python {venv_q} {shlex.quote(remote)}; "
            result = await run_shielded(runtime.run(["sh", "-c", setup], {}))
            if result.exit_code != 0:
                raise ToolsetError(
                    f"server {server.server_name!r} install failed in runtime: "
                    f"{(result.stderr or result.stdout).strip()[-2000:]}"
                )
            runtime.backend.cache.mcp_sources.update(pending)
        if target.user is not None:
            # Root's uv lives in its home; copy it into the venv, open to the user.
            python_q = shlex.quote(python_dir)
            share = (
                'set -e; export PATH="$HOME/.local/bin:$PATH"; '
                f'cp "$(command -v uv)" {venv_q}/bin/uv; chmod -R a+rX {venv_q}; '
                f"[ ! -d {python_q} ] || chmod -R a+rX {python_q}"
            )
            result = await run_shielded(runtime.run(["sh", "-c", share], {}))
            if result.exit_code != 0:
                raise ToolsetError(
                    f"server {server.server_name!r} could not be shared with "
                    f"{target.user!r}: {(result.stderr or result.stdout).strip()[-2000:]}"
                )
    return f"{venv}/bin/python"


async def log_tail(runtime: Runtime, log: str, limit: int = 2000) -> str:
    if limit <= 0:
        return ""
    with contextlib.suppress(Exception):
        # Tail in place so a large remote log never crosses into host memory in full.
        result = await runtime.run(["tail", "-c", str(limit), log], {})
        if result.exit_code == 0:
            return result.stdout
    return ""


async def _read_back_port(runtime: Runtime, path: str, deadline: float) -> int:
    """Poll the server's port file until the server writes it, often at first: a
    local server writes it within a fraction of a second."""
    loop = asyncio.get_running_loop()
    delay = 0.05
    while True:
        with contextlib.suppress(Exception):
            data = (await runtime.read(path)).decode().strip()
            if data.isdigit():
                return int(data)
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise ToolsetError(
                f"server did not report its port at {path} in its runtime"
            )
        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 2, 1)


async def serve_in_runtime(
    server: ServerBase,
    runtime: Runtime,
    *,
    exposed: bool,
    listen_all: bool = False,
    state_url: str | None = None,
    state_secret: str = "",
) -> int:
    """Start a server and return its bound port.

    An `exposed` server binds the port a local container publishes. Any other lets the OS
    choose and reports it through a file. With a state channel, the server fetches
    the current rollout task from the adjacent `/task` endpoint rather than a launch argument.
    """
    # A shared server has a private service secret but no fixed state URL. Set
    # both controls explicitly so a subprocess cannot inherit stale host values.
    env = {
        "VF_CONFIG": server.config.model_dump_json(),
        "VF_STATE_URL": state_url or "",
        "VF_STATE_SECRET": state_secret,
    }
    if runtime.type == "subprocess":
        # Keep provider temp files in the runtime workdir so cleanup removes them.
        assert runtime.info.id is not None
        env["TMPDIR"] = runtime.info.id
    if listen_all or (exposed and runtime.published_port is not None):
        env["MCP_HOST"] = "0.0.0.0"
    fixed = runtime.published_port if exposed else None
    port_file = None
    if fixed is not None:
        env["MCP_PORT"] = str(fixed)
    else:
        port_file = f"/tmp/vf-port-{uuid.uuid4().hex}"
        env["MCP_PORT_FILE"] = port_file
    python = sys.executable
    if runtime.type != "subprocess":
        python = await _install_in_sandbox(server, runtime)
    command = [python, "-m", type(server).__module__]
    if runtime.type != "subprocess":
        # Providers may invoke uv after the install shell exits, so preserve its PATH.
        path = "$HOME/.local/bin:$PATH"
        if runtime.user is not None:
            path = f"{PurePosixPath(python).parent}:{path}"
        command = [
            "sh",
            "-c",
            f'export PATH="{path}"; exec {shlex.join(command)}',
        ]
    log = f"vf_tool_{server.server_name}.log"
    if runtime.user is not None:
        # The agent user may not be able to write the workdir.
        log = f"/tmp/vf_tool_{server.server_name}-{uuid.uuid4().hex[:8]}.log"
    loop = asyncio.get_running_loop()
    # Servers without a toolset config, such as custom `ServerBase` subclasses, keep the default.
    timeouts = (
        server.config.timeout
        if isinstance(server.config, ToolsetConfig)
        else ToolsetTimeoutConfig()
    )
    timeout = timeouts.startup
    deadline = loop.time() + (math.inf if timeout is None else timeout)
    await runtime.run_background(command, env, log)
    if fixed is not None:
        port = fixed
    else:
        try:
            port = await _read_back_port(runtime, port_file, deadline)
        except ToolsetError as e:
            raise ToolsetError(f"{e}: {await log_tail(runtime, log)}") from e
    probe = await runtime.run(
        [
            python,
            "-c",
            _PROBE,
            f"http://127.0.0.1:{port}/mcp",
            str(max(deadline - loop.time(), 1)),
        ],
        {},
    )
    if probe.exit_code != 0:
        raise ToolsetError(
            f"tool server {server.server_name!r} not serving in runtime after "
            f"{timeout}s (raise its timeout.startup): {await log_tail(runtime, log)}"
        )
    return port


@contextlib.asynccontextmanager
async def _serve_local(
    server: ServerBase, state_url: str | None, state_secret: str
) -> AsyncIterator[tuple[str, Runtime]]:
    """Run `server` in a subprocess or Docker runtime of its own; yield its MCP URL on
    the evaluator's loopback and that runtime."""
    async with provision_backend(server.config.runtime) as backend:
        runtime = backend.bind()
        port = await serve_in_runtime(
            server,
            runtime,
            exposed=True,
            state_url=runtime.host_url(state_url) if state_url else None,
            state_secret=state_secret,
        )
        yield f"{(await runtime.expose(port)).rstrip('/')}/mcp", runtime


@contextlib.asynccontextmanager
async def launch_tools(
    toolsets: list[Toolset], *, state_secret: str, state_base: str | None
) -> AsyncIterator[dict[str, str]]:
    """Start the rollout's own servers that serve on the evaluator's loopback, all at
    once, and yield their loopback MCP URLs by name (`serve_tools` reaches them). They
    need nothing from the workspace, so a rollout starts them while its workspace
    boots."""
    state_url = f"{state_base.rstrip('/')}/state" if state_base else None
    local = [
        toolset
        for toolset in toolsets
        if not toolset.config.url
        and not toolset.config.shared
        and toolset.config.service is None
        and not isinstance(toolset.config.runtime, PrimeConfig)
    ]
    async with contextlib.AsyncExitStack() as stack:
        try:
            async with asyncio.TaskGroup() as group:
                started = {
                    toolset.server_name: group.create_task(
                        stack.enter_async_context(
                            _serve_local(toolset, state_url, state_secret)
                        )
                    )
                    for toolset in local
                }
        except ExceptionGroup as failed:
            raise failed.exceptions[0]  # one server's failure, typed for retries
        yield {name: task.result()[0] for name, task in started.items()}


@dataclass(frozen=True)
class SharedToolServer:
    """One live eval-scoped server, as rollouts see it: `url` is on the evaluator's
    loopback, or on the eval LAN `lan` that a Prime workspace joins to reach it.
    `runtime` translates a rollout's state URL into the server's network, and
    `state_secret` authenticates the server's state calls."""

    url: str
    runtime: Runtime = field(repr=False)
    lan: str | None = None
    state_secret: str = field(default="", repr=False)


@contextlib.asynccontextmanager
async def serve_shared(
    toolsets: list[Toolset],
) -> AsyncIterator[dict[str, SharedToolServer]]:
    """Start `shared` tool servers once for a whole eval and yield them by name. A
    server on a Prime VM joins a LAN of the eval's own; any other serves on the
    evaluator's loopback. A shared server is task-agnostic: it runs `setup()` once and
    receives each rollout's state channel per call."""
    servers: dict[str, SharedToolServer] = {}
    vms = {
        toolset.server_name: toolset.config.runtime
        for toolset in toolsets
        if isinstance(toolset.config.runtime, PrimeConfig)
    }
    async with contextlib.AsyncExitStack() as stack:
        lan, backends = (
            await stack.enter_async_context(eval_lan(vms)) if vms else (None, {})
        )
        for toolset in toolsets:
            name = toolset.server_name
            if name in servers:
                raise ToolsetError(
                    f"duplicate shared tool server name {name!r}; give one a distinct "
                    "TOOL_PREFIX"
                )
            if type(toolset).setup_task is not ServerBase.setup_task:
                logger.warning(
                    "shared server %r overrides `setup_task`, which a shared server "
                    "never runs; move task-agnostic work into `setup`, or unset "
                    "`shared` to start it per rollout.",
                    name,
                )
            state_secret = (
                secrets.token_urlsafe(24) if toolset._state_cls is not State else ""
            )
            if name in backends:
                runtime = backends[name].bind()
                port = await serve_in_runtime(
                    toolset,
                    runtime,
                    exposed=False,
                    listen_all=True,
                    state_secret=state_secret,
                )
                url = f"http://{lan_host(backends[name])}:{port}/mcp"
            else:
                url, runtime = await stack.enter_async_context(
                    _serve_local(toolset, None, state_secret)
                )
            servers[name] = SharedToolServer(url, runtime, lan, state_secret)
            logger.info("shared tool server %r: %s", name, url)
        yield servers


def join_shared_lans(
    toolsets: list[Toolset],
    shared: Mapping[str, SharedToolServer],
    workspace: Runtime,
) -> asyncio.Future[list[None]]:
    """Start joining `workspace` to the eval LANs of the shared servers among
    `toolsets`, which `serve_tools` then reaches by LAN name. A join takes a second or
    two, so the caller overlaps it with its own setup; a workspace that cannot join
    fails here, at once."""
    hosts: dict[str, set[str]] = {}
    for toolset in toolsets:
        live = shared.get(toolset.server_name) if toolset.config.shared else None
        if live is not None and live.lan is not None:
            hosts.setdefault(live.lan, set()).add(urlsplit(live.url).hostname or "")
    if hosts and not isinstance(workspace.backend, PrimeBackend):
        raise ToolsetError(
            "shared tool servers on Prime VMs are reached only from a Prime "
            f"workspace (not {workspace.type})"
        )
    return asyncio.gather(
        *(join_lan(workspace, lan, names) for lan, names in hosts.items())
    )


def _shared_url_for_rollout(
    server: SharedToolServer,
    visible_url: str,
    state_base: str | None,
    state_route: str,
) -> str:
    """Attach signed state coordinates; the shared server keeps its bearer private."""
    if not state_base or not server.state_secret:
        return visible_url
    state_url = server.runtime.host_url(f"{state_base.rstrip('/')}/state")
    parts = urlsplit(visible_url)
    query = dict(parse_qsl(parts.query))
    query[STATE_URL_PARAM] = state_url
    query[STATE_ROUTE_PARAM] = state_route
    query[STATE_SIGNATURE_PARAM] = state_signature(
        server.state_secret, state_url, state_route
    )
    return urlunsplit(parts._replace(query=urlencode(query)))


@contextlib.asynccontextmanager
async def serve_tools(
    toolsets: list[Toolset],
    workspace: Runtime,
    deployment: Deployment,
    shared: Mapping[str, SharedToolServer],
    local: Mapping[str, str],
    *,
    state_secret: str = "",
    state_route: str = "",
    state_base: str | None = None,
    proxy_mcp: MCPProxy | None = None,
):
    """Bring up a rollout's tool servers and yield `{name: url}` as `workspace`
    reaches them. A `shared` server is already running (see `serve_shared`) and joins
    under signed per-rollout state coordinates; one on an eval LAN needs the workspace
    to have joined it (`join_shared_lans`). The rollout's servers on the evaluator are
    already running at their `local` URLs (see `launch_tools`); one in a task service
    or on a Prime VM of its own is launched here. Each receives the private
    `state_secret` and fetches its task over the interception's `/task` channel.
    `state_base` is the interception's URL as reachable from any runtime."""
    urls: dict[str, str] = {}
    state_url = f"{state_base.rstrip('/')}/state" if state_base else None
    async with contextlib.AsyncExitStack() as stack:
        for toolset in toolsets:
            name, cfg = toolset.server_name, toolset.config
            if name in urls:
                raise ToolsetError(
                    f"duplicate tool server name {name!r}; give one a distinct "
                    "TOOL_PREFIX"
                )
            if cfg.url:
                urls[name] = workspace.host_url(cfg.url)
                logger.info("tool server %r (remote): %s", name, cfg.url)
                continue
            live = shared.get(name) if cfg.shared else None
            if cfg.shared and live is None:
                raise ToolsetError(
                    f"shared tool server {name!r} is not running; run the task through "
                    "an Env, or start it with `serve_shared` and pass it as `tools`"
                )
            if live is not None and live.lan is not None:
                url = live.url
            elif cfg.service is not None or isinstance(cfg.runtime, PrimeConfig):
                # A task service, or a Prime VM of the server's own on the rollout's LAN.
                target = cfg.service or name
                if target not in deployment.target_backends:
                    raise ToolsetError(
                        f"tool server {name!r} runs in service {target!r}, which the "
                        "task's services do not define"
                        if cfg.service
                        else f"tool server {name!r} runs on a Prime VM of its own, "
                        "which needs a Prime workspace the rollout provisions itself"
                    )
                runtime = deployment.target(target)
                port = await serve_in_runtime(
                    toolset,
                    runtime,
                    exposed=False,
                    listen_all=True,
                    state_url=runtime.host_url(state_url) if state_url else None,
                    state_secret=state_secret,
                )
                url = f"http://{deployment.hosts.get(target, target)}:{port}/mcp"
            else:
                # On the evaluator's loopback: a local workspace reaches it directly,
                # a remote one through its interception.
                url = live.url if live is not None else local[name]
                if workspace.is_local:
                    url = workspace.host_url(url)
                else:
                    assert proxy_mcp is not None
                    url = await stack.enter_async_context(proxy_mcp(url))
            if live is not None:
                url = _shared_url_for_rollout(live, url, state_base, state_route)
            urls[name] = url
            logger.info("tool server %r ready", name)
        yield urls
