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
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import PathDistribution, distribution, packages_distributions
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import ToolsetError
from verifiers.v1.interception.tunnel import PrimeTunnel
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
    backend_cls,
    make_backend,
    provision_backend,
)
from verifiers.v1.runtimes.base import _ENSURE_UV, SERVICE_PORT
from verifiers.v1.runtimes.private_network import (
    WORKSPACE_TAG,
    keepalive,
    lan_host,
    lan_network,
)
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
    time.sleep(min(1, remaining))
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


async def _install_in_sandbox(server: ServerBase, runtime: Runtime) -> str:
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
    root_q, temp_q, cache_q, venv_q = map(shlex.quote, (root, temp, cache, venv))
    # Servers in the same service share one environment. Serialize installs and
    # only remember sources after a successful install.
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
    """Poll the server's port file until the server writes it."""
    loop = asyncio.get_running_loop()
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
        await asyncio.sleep(min(1, remaining))


async def serve_in_runtime(
    server: ServerBase,
    runtime: Runtime,
    *,
    exposed: bool,
    listen_all: bool = False,
    port: int | None = None,
    state_url: str | None = None,
    state_secret: str = "",
) -> int:
    """Start a server and return its bound port.

    Exposed remote servers must use the runtime's forwarded port. Local servers let
    the OS choose and report the result through a file. With a state channel, the server fetches
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
    fixed = port if port is not None else runtime.published_port if exposed else None
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
        command = [
            "sh",
            "-c",
            f'export PATH="$HOME/.local/bin:$PATH"; exec {shlex.join(command)}',
        ]
    log = f"vf_tool_{server.server_name}.log"
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
async def reachable_url(
    service_runtime: Runtime,
    port: int,
    *,
    consumer_is_local: bool,
) -> AsyncIterator[str]:
    """Publish a server's port, tunneling host-local URLs for remote consumers."""
    url = await service_runtime.expose(port)
    if service_runtime.is_local and not consumer_is_local:
        async with PrimeTunnel().expose(urlsplit(url).port or 80) as public:
            yield public
    else:
        yield url


@dataclass(frozen=True)
class _ServedServer:
    url: str
    runtime: Runtime
    port: int


@contextlib.asynccontextmanager
async def _serve(
    server: ServerBase,
    harness_runtime: Runtime | None = None,
    harness_is_local: bool = True,
    *,
    state_secret: str = "",
    state_base: str | None = None,
    proxy_mcp: MCPProxy | None = None,
    runtime: Runtime | None = None,
    host: str | None = None,
    port: int | None = None,
):
    cfg = server.config
    if getattr(cfg, "service", None) is not None and runtime is None:
        raise ToolsetError(
            f"server {server.server_name!r} needs its service runtime from the deployment"
        )
    async with contextlib.AsyncExitStack() as stack:
        # Filtered servers need a LAN and policy enforcement owned by the deployment
        # or shared serving context; public port exposure cannot provide that path.
        if (
            isinstance(cfg.runtime, NetworkPolicyConfig)
            and cfg.runtime.network_restricted
            and runtime is None
        ):
            raise ToolsetError(
                f"server {server.server_name!r} restricts its runtime's network, which "
                "needs a Prime runtime of its own beside a Prime workspace "
                "so it can join the private network"
            )
        if runtime is None and not backend_cls(cfg.runtime).SUPPORTS_PORT_EXPOSURE:
            raise ToolsetError(
                f"server {server.server_name!r} runs in its own runtime, but the "
                f"{cfg.runtime.type} runtime cannot expose a port; use a supported runtime"
            )
        if runtime is None:
            backend = await stack.enter_async_context(provision_backend(cfg.runtime))
            runtime = backend.bind()
        # The shared-state channel: every server reaches the interception at the rollout's
        # `state_base`, which is universally reachable (the interception is exposed via a tunnel
        # whenever any consumer is remote). Eval-level shared servers get no per-rollout channel
        # (`state_base` is None for them).
        state_url = (
            runtime.host_url(f"{state_base.rstrip('/')}/state") if state_base else None
        )
        port = await serve_in_runtime(
            server,
            runtime,
            exposed=host is None,
            listen_all=host is not None,
            port=port,
            state_url=state_url,
            state_secret=state_secret,
        )
        if host is not None:
            yield _ServedServer(f"http://{host}:{port}/mcp", runtime, port)
            return
        consumer_is_local = (
            harness_runtime.is_local
            if harness_runtime is not None
            else harness_is_local
        )
        if proxy_mcp is not None and runtime.is_local and not consumer_is_local:
            backend = await runtime.expose(port)
            # Register after startup, and revoke/drain before releasing the runtime.
            url = await stack.enter_async_context(
                proxy_mcp(f"{backend.rstrip('/')}/mcp")
            )
            yield _ServedServer(url, runtime, port)
            return
        base = await stack.enter_async_context(
            reachable_url(runtime, port, consumer_is_local=consumer_is_local)
        )
        if harness_runtime is not None:
            base = harness_runtime.host_url(base)
        yield _ServedServer(f"{base.rstrip('/')}/mcp", runtime, port)


@contextlib.asynccontextmanager
async def serve(
    server: ServerBase,
    harness_runtime: Runtime | None = None,
    harness_is_local: bool = True,
    *,
    state_secret: str = "",
    state_base: str | None = None,
    proxy_mcp: MCPProxy | None = None,
    runtime: Runtime | None = None,
    host: str | None = None,
):
    """Serve one MCP server and yield the URL visible to its consumer."""
    async with _serve(
        server,
        harness_runtime,
        harness_is_local,
        state_secret=state_secret,
        state_base=state_base,
        proxy_mcp=proxy_mcp,
        runtime=runtime,
        host=host,
    ) as served:
        yield served.url


@dataclass(frozen=True)
class SharedToolServer:
    """Serializable connection details borrowed from a shared serving owner.

    Workers receive no runtime or cleanup handles. External endpoints do not
    participate in the framework's state channel.
    """

    url: str
    local: bool
    external: bool = False
    network_id: str | None = None
    lan_url: str | None = None
    definition: tuple[str, str, str] = ("", "", "")
    state_secret: str = field(default="", repr=False)


@contextlib.asynccontextmanager
async def serve_shared(
    toolsets: list[Toolset],
    harness_is_local: bool = True,
    *,
    harness_is_prime: bool = False,
):
    """Own shared servers and their LAN once; lend portable handles to all workers."""
    servers: dict[str, SharedToolServer] = {}
    async with contextlib.AsyncExitStack() as stack:
        names = [toolset.server_name for toolset in toolsets]
        if len(set(names)) != len(names):
            raise ToolsetError("duplicate shared tool server name in Task.toolsets")
        for toolset in toolsets:
            if (
                not toolset.config.url
                and type(toolset).setup_task is not ServerBase.setup_task
            ):
                raise ToolsetError(
                    f"shared server {toolset.server_name!r} overrides setup_task; "
                    "use setup for shared resources and self.state for rollout state"
                )
        network_id = None
        if any(
            isinstance(t.config.runtime, PrimeConfig) and not t.config.url
            for t in toolsets
        ):
            from prime_sandboxes import LanRule

            network_id = await stack.enter_async_context(
                lan_network(
                    [
                        LanRule(
                            from_=f"tag:{WORKSPACE_TAG}",
                            to="tag:vf-tool",
                            ports=[SERVICE_PORT],
                        )
                    ]
                )
            )

        async def start(toolset):
            cfg = toolset.config
            name = toolset.server_name
            definition = (
                type(toolset).__module__,
                type(toolset).__qualname__,
                cfg.model_dump_json(),
            )
            if cfg.url:  # already running remotely; nothing launched, nothing to bridge
                servers[name] = SharedToolServer(
                    url=cfg.url, local=False, external=True, definition=definition
                )
            else:
                runtime = None
                if isinstance(cfg.runtime, PrimeConfig):
                    backend = make_backend(cfg.runtime)
                    backend.lan = (network_id, ["vf-tool"])
                    stack.push_async_callback(backend.stop)
                    await backend.start()
                    await stack.enter_async_context(keepalive(backend))
                    runtime = backend.bind()
                state_secret = (
                    secrets.token_urlsafe(24) if toolset._state_cls is not State else ""
                )
                served = await stack.enter_async_context(
                    _serve(
                        toolset,
                        harness_is_local=harness_is_local,
                        state_secret=state_secret,
                        runtime=runtime,
                        port=SERVICE_PORT if runtime is not None else None,
                        host=lan_host(runtime.backend)
                        if runtime is not None and harness_is_prime
                        else None,
                    )
                )
                if runtime is not None:
                    await runtime.backend.prepare_execution([])
                servers[name] = SharedToolServer(
                    url=served.url,
                    local=served.runtime.is_local,
                    network_id=network_id if runtime is not None else None,
                    lan_url=f"http://{lan_host(runtime.backend)}:{served.port}/mcp"
                    if runtime is not None
                    else None,
                    definition=definition,
                    state_secret=state_secret,
                )
            logger.info("shared tool server '%s': %s", name, servers[name].url)

        async with asyncio.TaskGroup() as group:
            for toolset in toolsets:
                group.create_task(start(toolset))
        yield {name: servers[name] for name in names}


def _shared_url_for_rollout(
    server: SharedToolServer,
    visible_url: str,
    state_base: str | None,
    state_route: str,
) -> str:
    """Attach signed state coordinates; the shared server keeps its bearer private."""
    if not state_base or not server.state_secret:
        return visible_url
    state_url = f"{state_base.rstrip('/')}/state"
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
    harness_runtime: Runtime,
    shared: dict[str, SharedToolServer] | None = None,
    *,
    state_secret: str = "",
    state_route: str = "",
    state_base: str | None = None,
    proxy_mcp: MCPProxy | None = None,
    deployment: Deployment | None = None,
):
    """Resolve each declared server from its owner and yield the harness's MCP URLs."""
    urls: dict[str, str] = {}
    names = [toolset.server_name for toolset in toolsets]
    if len(set(names)) != len(names):
        raise ToolsetError("duplicate tool server name in Task.toolsets")
    async with contextlib.AsyncExitStack() as stack:

        async def start(toolset):
            name = toolset.server_name
            cfg = toolset.config
            if cfg.shared:
                server = (shared or {}).get(name)
                if server is None:
                    raise ToolsetError(
                        f"shared server {name!r} needs a serving owner; use Env.serving() or serve_shared()"
                    )
                definition = (
                    type(toolset).__module__,
                    type(toolset).__qualname__,
                    cfg.model_dump_json(),
                )
                if server.definition != definition:
                    raise ToolsetError(
                        f"shared server {name!r} differs from its serving owner's declaration"
                    )
                url = (
                    server.lan_url
                    if isinstance(harness_runtime.backend, PrimeBackend)
                    and server.lan_url
                    else harness_runtime.host_url(server.url)
                )
                urls[name] = (
                    url
                    if server.external
                    else _shared_url_for_rollout(server, url, state_base, state_route)
                )
            elif cfg.url:
                urls[name] = harness_runtime.host_url(cfg.url)
            else:
                runtime, host = (
                    deployment.tool_target(toolset)
                    if deployment is not None
                    else (None, None)
                )
                if cfg.service is not None and runtime is None:
                    raise ToolsetError(
                        f"server {name!r} needs a deployment whose services define {cfg.service!r}"
                    )
                urls[name] = await stack.enter_async_context(
                    serve(
                        toolset,
                        harness_runtime,
                        state_secret=state_secret,
                        state_base=state_base,
                        proxy_mcp=proxy_mcp,
                        runtime=runtime,
                        host=host,
                    )
                )
            logger.info("tool server '%s' ready", name)

        try:
            async with asyncio.TaskGroup() as group:
                for toolset in toolsets:
                    group.create_task(start(toolset))
        except ExceptionGroup as failed:
            raise failed.exceptions[0]
        yield {name: urls[name] for name in names}
