from __future__ import annotations

import asyncio
import contextlib
import math
import secrets
import shlex
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import distribution, packages_distributions
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import TypeAdapter

from verifiers.v1.configs.runtime import CustomRuntimeConfig, NetworkPolicyConfig
from verifiers.v1.errors import ToolsetError
from verifiers.v1.mcp.auth import MCPURL
from verifiers.v1.mcp.packages import install_package
from verifiers.v1.mcp.server import (
    STATE_ROUTE_PARAM,
    STATE_SIGNATURE_PARAM,
    STATE_URL_PARAM,
    ServerBase,
    state_signature,
)
from verifiers.v1.mcp.toolset import BaseToolsetConfig, ToolsetTimeoutConfig
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    provision_backend,
    runtime_is_local,
)
from verifiers.v1.runtimes.base import _ENSURE_UV, SERVICE_PORT
from verifiers.v1.state import State
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import get_logger

if TYPE_CHECKING:
    from verifiers.v1.mcp.toolset import Toolset

logger = get_logger(__name__)

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
def _package_dir(cls: Callable) -> str:
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


async def _install_in_sandbox(server: ServerBase, agent_runtime: Runtime) -> str:
    return await install_in_runtime(
        type(server), agent_runtime, label=f"server {server.server_name!r}"
    )


async def install_in_runtime(
    cls: Callable, agent_runtime: Runtime, *, label: str
) -> str:
    """Install the framework and a component's source package in a runtime."""
    # Dependency installation is trusted setup as the backend's default user.
    # The server itself keeps the agent's user when launched.
    runtime = agent_runtime.backend.bind(env=agent_runtime.env).with_user(None)
    # Prime mounts /tmp as a small tmpfs. Keep dependencies on the root disk,
    # outside the task workspace so its files contain only the task's work.
    support = "/var/tmp/tool-service"
    root = f"{support}/src"
    temp = f"{support}/tmp"
    cache = f"{support}/uv-cache"
    venv = f"{support}/venv"
    python_dir = f"{support}/python"
    root_q, temp_q, cache_q, venv_q = map(shlex.quote, (root, temp, cache, venv))
    # Servers in the same service install into one physical environment.
    # Serialize its mutations and only remember sources after a successful install.
    async with runtime.backend.cache.mcp_lock:
        sources = dict.fromkeys((_package_dir(ServerBase), _package_dir(cls)))
        pending = [
            source
            for source in sources
            if source not in runtime.backend.cache.mcp_sources
        ]
        if pending:
            env = {"TMPDIR": temp, "UV_CACHE_DIR": cache}
            if agent_runtime.user is not None:
                env["UV_PYTHON_INSTALL_DIR"] = python_dir
            setup = (
                f"set -e; mkdir -p {root_q} {temp_q} {cache_q}; "
                'export PATH="$HOME/.local/bin:$PATH"; '
            )
            if not runtime.backend.cache.mcp_sources:
                # Failed installs can leave the venv behind; retain it when retrying.
                setup += f"{_ENSURE_UV}; uv venv --allow-existing {venv_q}; "
            result = await run_shielded(runtime.run(["sh", "-c", setup], env))
            if result.exit_code != 0:
                raise ToolsetError(
                    f"{label} environment setup failed in runtime: "
                    f"{(result.stderr or result.stdout).strip()[-2000:]}"
                )
            for source in pending:
                await install_package(
                    runtime, Path(source), root, env, python=f"{venv}/bin/python"
                )
                runtime.backend.cache.mcp_sources.add(source)
        if agent_runtime.user is not None:
            # Root's uv lives in its home; copy it into the venv, open to the agent.
            python_q = shlex.quote(python_dir)
            share = (
                'set -e; export PATH="$HOME/.local/bin:$PATH"; '
                f'cp "$(command -v uv)" {venv_q}/bin/uv; chmod -R a+rX {venv_q}; '
                f"[ ! -d {python_q} ] || chmod -R a+rX {python_q}"
            )
            result = await run_shielded(runtime.run(["sh", "-c", share], {}))
            if result.exit_code != 0:
                raise ToolsetError(
                    f"{label} could not be shared with "
                    f"{agent_runtime.user!r}: {(result.stderr or result.stdout).strip()[-2000:]}"
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
    state_url: str | None = None,
    state_secret: str = "",
    mcp_secret: str,
) -> int:
    """Start a server and return its bound port.

    Exposed servers bind the runtime's published `SERVICE_PORT`. Otherwise, the OS
    chooses a port and reports it through a file. With a state channel, the server
    fetches the current rollout task from the adjacent `/task` endpoint.
    """
    # A shared server has a private service secret but no fixed state URL. Set
    # both controls explicitly so a subprocess cannot inherit stale host values.
    env = {
        "VF_CONFIG": server.config.model_dump_json(),
        "VF_STATE_URL": state_url or "",
        "VF_STATE_SECRET": state_secret,
        "TOOL_SERVICE_MCP_SECRET": mcp_secret,
    }
    if listen_all or exposed:
        env["MCP_HOST"] = "0.0.0.0"
    fixed = SERVICE_PORT if exposed else None
    port_file = None
    if fixed is not None:
        env["MCP_PORT"] = str(fixed)
    else:
        port_file = runtime.temp_path(f"tool-port-{uuid.uuid4().hex}", env=env)
        env["MCP_PORT_FILE"] = port_file
    python = await _install_in_sandbox(server, runtime)
    command = [python, "-m", type(server).__module__]
    # Providers may invoke uv after the install shell exits, so preserve its PATH.
    path = "$HOME/.local/bin:$PATH"
    if runtime.user is not None:
        path = f"{PurePosixPath(python).parent}:{path}"
    command = [
        "sh",
        "-c",
        f'export PATH="{path}"; exec {shlex.join(command)}',
    ]
    # Keep infrastructure logs out of task artifacts and support read-only workdirs.
    log = runtime.temp_path(f"tool-{uuid.uuid4().hex}.log", env=env)
    loop = asyncio.get_running_loop()
    # Servers without a toolset config, such as custom `ServerBase` subclasses, keep the default.
    timeouts = (
        server.config.timeout
        if isinstance(server.config, BaseToolsetConfig)
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


@dataclass(frozen=True)
class _ServedServer:
    url: str
    runtime: Runtime
    mcp_secret: str


@contextlib.asynccontextmanager
async def _serve(
    server: ServerBase,
    provider: RuntimeConfig,
    harness_runtime: Runtime | None = None,
    *,
    state_secret: str = "",
    state_base: str | None = None,
    services: Mapping[str, Runtime] | None = None,
    hosts: Mapping[str, str] | None = None,
    placements: Mapping[str, Runtime] | None = None,
):
    """Serve `server` in a service, a VM on the rollout's private network, or a runtime
    of its own (`tool_runtime` over the harness's `provider`), and yield its URL as the
    harness reaches it."""
    cfg = server.config
    service = getattr(cfg, "service", None)
    own_vm = (placements or {}).get(server.server_name)
    mcp_secret = secrets.token_urlsafe(32)
    async with contextlib.AsyncExitStack() as stack:
        host = None
        if service is not None or own_vm is not None:
            if own_vm is not None:
                runtime = own_vm
                host = (hosts or {})[server.server_name]
            elif services is None or service not in services:
                raise ToolsetError(
                    f"server {server.server_name!r} runs in service {service!r}, "
                    "which needs a task whose services define it"
                )
            else:
                runtime = services[service]
                host = (hosts or {}).get(service, service)
            if runtime.is_local:
                raise ToolsetError(
                    f"server {server.server_name!r} runs in service {service!r}, "
                    "which needs a native Prime service VM"
                )
        else:
            config = tool_runtime(cfg, provider)
            if isinstance(config, NetworkPolicyConfig) and config.network_restricted:
                raise ToolsetError(
                    f"server {server.server_name!r} restricts its runtime's network, "
                    "which needs a Prime runtime of its own beside a Prime workspace "
                    "without a task service spec (it then joins the rollout's "
                    "private network)"
                )
            if harness_runtime is not None:
                check_reachable(server.server_name, config, harness_runtime.is_local)
            backend = await stack.enter_async_context(provision_backend(config))
            runtime = backend.bind()

        port = await serve_in_runtime(
            server,
            runtime,
            exposed=host is None,
            listen_all=host is not None,
            state_url=(
                runtime.host_url(f"{state_base.rstrip('/')}/state")
                if state_base
                else None
            ),
            state_secret=state_secret,
            mcp_secret=mcp_secret,
        )
        if host is not None:
            yield _ServedServer(f"http://{host}:{port}/mcp", runtime, mcp_secret)
            return
        base = await runtime.expose(port)
        if harness_runtime is not None:
            base = harness_runtime.host_url(base)
        yield _ServedServer(f"{base.rstrip('/')}/mcp", runtime, mcp_secret)


def tool_runtime(config: BaseToolsetConfig, provider: RuntimeConfig) -> RuntimeConfig:
    """The runtime a launched server runs in: its configured one, or by default a
    sandbox of its own from the harness's `provider` (a Docker container beside a Docker
    harness, a Prime VM beside a Prime one)."""
    if config.runtime is not None:
        return config.runtime
    data: dict = {"type": provider.type}
    if isinstance(provider, CustomRuntimeConfig):
        data["import_path"] = provider.import_path
    return TypeAdapter(RuntimeConfig).validate_python(data)


def check_reachable(name: str, config: RuntimeConfig, harness_is_local: bool) -> None:
    """Refuse a server on a local runtime for a remote harness: a Prime sandbox cannot
    reach a server on this machine."""
    if runtime_is_local(config) and not harness_is_local:
        raise ToolsetError(
            f"tool server {name!r} runs on the local {config.type} runtime, which a "
            "remote harness cannot reach; leave its runtime unset to run it on the "
            "harness's provider"
        )


@contextlib.asynccontextmanager
async def serve(
    server: ServerBase,
    harness_runtime: Runtime,
    *,
    state_secret: str = "",
    state_base: str | None = None,
    services: Mapping[str, Runtime] | None = None,
    hosts: Mapping[str, str] | None = None,
    placements: Mapping[str, Runtime] | None = None,
):
    """Serve one MCP server and yield the URL visible to its consumer."""
    async with _serve(
        server,
        harness_runtime.config,
        harness_runtime,
        state_secret=state_secret,
        state_base=state_base,
        services=services,
        hosts=hosts,
        placements=placements,
    ) as served:
        yield MCPURL(served.url, served.mcp_secret)


@dataclass(frozen=True)
class SharedToolServer:
    """One live taskset-scoped (shared) server, as the rollouts see it: its eval-level
    `url`, and the `runtime` it runs in. A remote runtime is an interception consumer,
    so the interception must be exposed for it to reach the `/state` channel (see
    `Env._requires_tunnel`); the runtime also translates that channel into the
    server's network. An `external` server (a config-`url` endpoint) was not launched
    by the framework and sits outside its state machinery entirely: rollouts get its
    URL bare — no state tag (and no per-rollout secret sent to a third party)."""

    url: str
    external: bool = False
    runtime: Runtime | None = field(default=None, repr=False)
    state_secret: str = field(default="", repr=False)
    mcp_secret: str = field(default="", repr=False)


@contextlib.asynccontextmanager
async def serve_shared(toolsets: list[Toolset], provider: RuntimeConfig):
    """Start the taskset-scoped (shared) tool servers ONCE for a whole eval, each in its OWN
    runtime (`tool_runtime` over the harnesses' `provider`), and yield
    `{name: SharedToolServer}` reachable by every rollout's harness. The caller
    (`Env.shared_tools`) checks that every harness can reach them. Torn down when the
    eval ends. A shared server is task-agnostic — the taskset carries no per-row data —
    so its `setup` gets no task (its `setup_task` is never called; the per-rollout
    servers fetch theirs over the `/task` channel)."""
    servers: dict[str, SharedToolServer] = {}
    async with contextlib.AsyncExitStack() as stack:
        for toolset in toolsets:
            cfg = toolset.config
            name = toolset.server_name
            if name in servers:
                raise ToolsetError(
                    f"duplicate shared tool server name '{name}' in Taskset.toolsets — "
                    f"give one a distinct TOOL_PREFIX"
                )
            if type(toolset).setup_task is not ServerBase.setup_task:
                logger.warning(
                    "shared server %r overrides `setup_task`, but `setup_task` is NEVER "
                    "called for a taskset-scoped server (it's built once, task-agnostic) — "
                    "its per-task logic will not run. Move task-agnostic work into `setup`, "
                    "or construct it in `Task.toolsets` to run it per-rollout.",
                    name,
                )
            if cfg.url:  # already running remotely; nothing launched
                servers[name] = SharedToolServer(url=cfg.url, external=True)
            else:
                state_secret = (
                    secrets.token_urlsafe(24) if toolset._state_cls is not State else ""
                )
                served = await stack.enter_async_context(
                    _serve(toolset, provider, state_secret=state_secret)
                )
                servers[name] = SharedToolServer(
                    url=served.url,
                    runtime=served.runtime,
                    state_secret=state_secret,
                    mcp_secret=served.mcp_secret,
                )
            logger.info("Started shared tool server %s url=%s", name, servers[name].url)
        yield servers


def _shared_url_for_rollout(
    server: SharedToolServer,
    visible_url: str,
    state_base: str | None,
    state_route: str,
) -> str:
    """Attach signed state coordinates; the shared server keeps its bearer private."""
    if not state_base or not server.state_secret:
        return MCPURL(visible_url, server.mcp_secret)
    state_url = f"{state_base.rstrip('/')}/state"
    if server.runtime is not None:
        state_url = server.runtime.host_url(state_url)
    parts = urlsplit(visible_url)
    query = dict(parse_qsl(parts.query))
    query[STATE_URL_PARAM] = state_url
    query[STATE_ROUTE_PARAM] = state_route
    query[STATE_SIGNATURE_PARAM] = state_signature(
        server.state_secret, state_url, state_route
    )
    return MCPURL(urlunsplit(parts._replace(query=urlencode(query))), server.mcp_secret)


@contextlib.asynccontextmanager
async def serve_tools(
    toolsets: list[Toolset],
    harness_runtime: Runtime,
    shared: dict[str, SharedToolServer] | None = None,
    *,
    state_secret: str = "",
    state_route: str = "",
    state_base: str | None = None,
    services: Mapping[str, Runtime] | None = None,
    hosts: Mapping[str, str] | None = None,
    placements: Mapping[str, Runtime] | None = None,
):
    """Bring up a rollout's tool servers and yield `{name: url}` the harness reaches: the
    task-scoped `toolsets` own their connections through `Toolset.serve`, and
    taskset-scoped `shared` servers, already running eval-level (see `serve_shared`),
    join under their per-rollout state tag.
    `state_secret` is private to task-scoped servers; shared servers keep an
    eval-level service secret and receive only signed `state_route` coordinates.
    `state_base` is universally reachable from either placement."""
    urls: dict[str, str] = {}
    async with contextlib.AsyncExitStack() as stack:
        for name, server in (shared or {}).items():
            # `host_url` translates a server on this machine; any other URL passes as is.
            url = harness_runtime.host_url(server.url)
            if server.external:
                # Not ours: a pre-existing endpoint with no vf state channel. Pass the URL
                # through bare — a state tag would be useless, and the per-rollout secret
                # must not ride the query string to a third-party host.
                urls[name] = url
                logger.debug("Using external tool server %s url=%s", name, server.url)
                continue
            urls[name] = _shared_url_for_rollout(server, url, state_base, state_route)
            logger.debug("Using shared tool server %s url=%s", name, server.url)
        for toolset in toolsets:
            name = toolset.server_name
            if name in urls:
                raise ToolsetError(
                    f"tool server name '{name}' is declared both taskset-scoped (shared) "
                    f"and task-scoped — pick one scope, or give one a distinct TOOL_PREFIX"
                )
            urls[name] = await stack.enter_async_context(
                toolset.serve(
                    harness_runtime,
                    state_secret=state_secret,
                    state_base=state_base,
                    services=services,
                    hosts=hosts,
                    placements=placements,
                )
            )
            logger.debug("Connected tool server %s", name)
        yield urls
