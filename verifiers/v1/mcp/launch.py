from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import shlex
import subprocess
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import cache
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
from verifiers.v1.runtimes import (
    Runtime,
    make_runtime,
)
from verifiers.v1.runtimes.base import _ENSURE_UV
from verifiers.v1.state import State

if TYPE_CHECKING:
    from verifiers.v1.mcp.toolset import Toolset

logger = logging.getLogger(__name__)

_SDIST_BUILD_TIMEOUT_SECONDS = 300


@dataclass
class _SdistBuildState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    build: asyncio.Task[tuple[str, bytes]] | None = None
    users: int = 0


# Coordination and caching are process-local. Spawned env-server workers each
# build once; within an event loop, concurrent rollouts share the completed artifact.
_SDIST_BUILD_STATES: dict[tuple[Path, asyncio.AbstractEventLoop], _SdistBuildState] = {}

# Any HTTP response, including MCP's 406 to a bare GET, proves the server is listening.
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


def _source_dir(cls: type) -> str | None:
    module = sys.modules.get(cls.__module__)
    path = getattr(module, "__file__", None)
    if not path:
        return None
    for parent in Path(path).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return str(parent)
    return None


@cache
def _build_sdist(src: Path) -> tuple[str, bytes]:
    """Build a project's source distribution through its declared PEP 517 backend."""
    with tempfile.TemporaryDirectory(prefix="vf-sdist-") as directory:
        out = Path(directory)
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
                timeout=_SDIST_BUILD_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as e:
            raise ToolsetError(
                f"cannot build source distribution for {src}: uv is not installed"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise ToolsetError(
                "source distribution build timed out after "
                f"{_SDIST_BUILD_TIMEOUT_SECONDS}s for {src}"
            ) from e
        if result.returncode != 0:
            detail = "\n".join(
                part.strip() for part in (result.stdout, result.stderr) if part.strip()
            )
            raise ToolsetError(
                f"source distribution build failed for {src}: {detail[-2000:]}"
            )
        artifacts = [path for path in out.iterdir() if path.is_file()]
        if len(artifacts) != 1:
            names = ", ".join(sorted(path.name for path in artifacts)) or "none"
            raise ToolsetError(
                f"build backend for {src} produced {len(artifacts)} source distributions: {names}"
            )
        artifact = artifacts[0]
        return artifact.name, artifact.read_bytes()


async def _cached_sdist(src: Path) -> tuple[str, bytes]:
    """Build once per source without parking waiters in the thread pool."""
    src = src.resolve()
    key = (src, asyncio.get_running_loop())
    state = _SDIST_BUILD_STATES.get(key)
    if state is None:
        state = _SdistBuildState()
        _SDIST_BUILD_STATES[key] = state
    state.users += 1
    try:
        async with state.lock:
            # Coordinate before entering the default executor so concurrent
            # same-source waiters do not occupy executor threads.
            if state.build is None:
                state.build = asyncio.create_task(asyncio.to_thread(_build_sdist, src))
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
        if state.users == 0 and _SDIST_BUILD_STATES.get(key) is state:
            del _SDIST_BUILD_STATES[key]


def _verifiers_root() -> Path:
    import verifiers

    root = Path(verifiers.__file__).resolve().parent.parent
    if not (root / "pyproject.toml").exists():
        raise ToolsetError(
            "verifiers is not a source checkout (no pyproject above the package), so it can't be "
            "uploaded to a sandbox; run sandboxed servers from a verifiers source install"
        )
    return root


async def _install_in_sandbox(server: ServerBase, runtime: Runtime) -> str:
    source_dir = _source_dir(type(server))
    if source_dir is None:
        raise ToolsetError(
            f"server {server.server_name!r} runs in a {runtime.type} runtime but its module is not "
            "a local package (no pyproject) — sandbox launch needs a local env package to upload"
        )
    # Prime VMs mount /tmp as a small tmpfs, while the runtime workdir lives on
    # the VM's root disk. Keep source, build scratch space, and uv's cache on the
    # durable runtime filesystem so ordinary dependency installs cannot exhaust
    # the tmpfs.
    workdir = str(PurePosixPath(runtime.config.workdir))
    root = str(PurePosixPath(workdir) / ".vf-src")
    temp = str(PurePosixPath(workdir) / ".vf-tmp")
    cache = str(PurePosixPath(workdir) / ".vf-uv-cache")
    vf, env = _verifiers_root(), Path(source_dir)
    vf_name, vf_data = await _cached_sdist(vf)
    if env == vf:
        env_name, env_data = vf_name, vf_data
    else:
        env_name, env_data = await _cached_sdist(env)
    vf_remote = f"{root}/{vf_name}"
    env_remote = f"{root}/{env_name}"
    await runtime.write(vf_remote, vf_data)
    if env_remote != vf_remote:
        await runtime.write(env_remote, env_data)
    venv = str(PurePosixPath(workdir) / ".vf-venv")
    root_q, temp_q, cache_q, venv_q = map(shlex.quote, (root, temp, cache, venv))
    extras = ",".join(type(server).EXTRAS)
    vf_source = shlex.quote(vf_remote)
    env_source = shlex.quote(env_remote + (f"[{extras}]" if extras else ""))
    setup = (
        f"set -e; mkdir -p {root_q} {temp_q} {cache_q}; "
        f"export TMPDIR={temp_q} UV_CACHE_DIR={cache_q}; "
        f"{_ENSURE_UV}; "
        f"uv venv {venv_q} && "
        f"uv pip install --python {venv_q} {vf_source} && "
        f"uv pip install --python {venv_q} {env_source}"
    )
    result = await runtime.run(["sh", "-c", setup], {})
    if result.exit_code != 0:
        raise ToolsetError(
            f"server {server.server_name!r} install failed in runtime: "
            f"{(result.stderr or result.stdout).strip()[-2000:]}"
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


async def _read_back_port(runtime: Runtime, path: str) -> int:
    """Poll the server's port file until the server writes it."""
    for _ in range(180):
        with contextlib.suppress(Exception):
            data = (await runtime.read(path)).decode().strip()
            if data.isdigit():
                return int(data)
        await asyncio.sleep(1)
    raise ToolsetError(f"server did not report its port at {path} in its runtime")


async def serve_in_runtime(
    server: ServerBase,
    runtime: Runtime,
    *,
    exposed: bool,
    state_url: str | None = None,
    state_secret: str = "",
) -> int:
    """Start a server and return its bound port.

    Exposed servers must use the runtime's forwarded port. Colocated servers let the OS choose and
    report the result through a file. With a state channel, the server fetches the current rollout
    task from the adjacent `/task` endpoint rather than a launch argument.
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
    if exposed and runtime.published_port is not None:
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
        # Prebuilt runtime images may already contain the server and its Python
        # environment. Let those servers bypass redundant per-launch source
        # uploads and installation while preserving the existing default.
        python = getattr(type(server), "RUNTIME_PYTHON", None)
        if python is None:
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
    await runtime.run_background(command, env, log)
    if fixed is not None:
        port = fixed
    else:
        try:
            port = await _read_back_port(runtime, port_file)
        except ToolsetError as e:
            raise ToolsetError(f"{e}: {await log_tail(runtime, log)}") from e
    probe = await runtime.run(
        [python, "-c", _PROBE, f"http://127.0.0.1:{port}/mcp"], {}
    )
    if probe.exit_code != 0:
        raise ToolsetError(
            f"tool server {server.server_name!r} not serving in runtime: {await log_tail(runtime, log)}"
        )
    return port


@contextlib.asynccontextmanager
async def reachable_url(
    service: Runtime, port: int, *, colocated: bool, consumer_is_local: bool
) -> AsyncIterator[str]:
    """Yield the URL a consumer uses to reach the server at (`service`, `port`), over two
    primitives: `Runtime.expose` (publish a port out of a sandbox) and a host `Tunnel` (reach
    into the host from a remote runtime). `colocated` = the server shares the consumer's
    runtime; `consumer_is_local` = the consumer can use a host-local URL without a tunnel.

    - `colocated` -> localhost (same runtime, in-sandbox or host loopback);
    - a non-colocated sandbox service -> its published URL (`expose`), when available;
    - a host-local URL -> direct for a local consumer, through a host tunnel for a remote one."""
    if colocated:
        yield f"http://127.0.0.1:{port}"
    elif published := await service.expose(port):
        if service.is_local and not consumer_is_local:
            published_port = urlsplit(published).port
            if published_port is None:
                raise ToolsetError(
                    f"{service.type} runtime exposed an invalid URL: {published}"
                )
            async with PrimeTunnel().expose(published_port) as url:
                yield url
        else:
            yield published
    else:
        raise ToolsetError(f"{service.type} runtime did not expose service port {port}")


@dataclass(frozen=True)
class _ServedServer:
    url: str
    runtime: Runtime


@contextlib.asynccontextmanager
async def _serve(
    server: ServerBase,
    harness_runtime: Runtime | None = None,
    harness_is_local: bool = True,
    *,
    state_secret: str = "",
    state_base: str | None = None,
):
    cfg = server.config
    colocated = getattr(cfg, "colocated", False)
    async with contextlib.AsyncExitStack() as stack:
        # Colocated servers inherit the harness cut. A separately provisioned filtered
        # server has neither that lifecycle nor a published port after isolation;
        # reject it instead of silently leaving its requested policy unenforced.
        if (
            isinstance(cfg.runtime, NetworkPolicyConfig)
            and cfg.runtime.network_restricted
            and not (colocated and harness_runtime is not None)
        ):
            raise ToolsetError(
                "Runtime network policies are supported on the harness runtime; "
                f"server {server.server_name!r} must be colocated or use an "
                "unrestricted runtime"
            )
        if colocated and harness_runtime is not None:
            runtime = harness_runtime
        else:
            runtime = make_runtime(cfg.runtime)
            await runtime.start()
            stack.push_async_callback(runtime.stop)
        # Only consumers outside the server runtime need its fixed published port. Colocated tools
        # use independent OS-assigned ports, avoiding clashes on the runtime's service port.
        exposed = runtime is not harness_runtime
        # The shared-state channel: every server reaches the interception at the rollout's
        # `state_base`, which is universally reachable (the interception is exposed via a tunnel
        # whenever any consumer is remote). Eval-level shared servers get no per-rollout channel
        # (`state_base` is None for them).
        state_url = (
            f"{runtime.host_url(state_base.rstrip('/'))}/state" if state_base else None
        )
        port = await serve_in_runtime(
            server,
            runtime,
            exposed=exposed,
            state_url=state_url,
            state_secret=state_secret,
        )
        # The harness consumes the server, and decides reachability: colocated when the
        # server shares the harness's runtime, reached with the harness's locality (read
        # off the harness runtime when there is one, else `harness_is_local` for an
        # eval-level shared tool).
        colocated = runtime is harness_runtime
        consumer_is_local = (
            harness_runtime.is_local
            if harness_runtime is not None
            else harness_is_local
        )
        base = await stack.enter_async_context(
            reachable_url(
                runtime, port, colocated=colocated, consumer_is_local=consumer_is_local
            )
        )
        if colocated and harness_runtime is not None and runtime.network_restricted:
            base = base.replace("127.0.0.1", "localhost", 1)
        elif not colocated and harness_runtime is not None:
            base = harness_runtime.host_url(base)
        yield _ServedServer(f"{base.rstrip('/')}/mcp", runtime)


@contextlib.asynccontextmanager
async def serve(
    server: ServerBase,
    harness_runtime: Runtime | None = None,
    harness_is_local: bool = True,
    *,
    state_secret: str = "",
    state_base: str | None = None,
):
    """Serve one MCP server and yield the URL visible to its consumer."""
    async with _serve(
        server,
        harness_runtime,
        harness_is_local,
        state_secret=state_secret,
        state_base=state_base,
    ) as served:
        yield served.url


@dataclass(frozen=True)
class SharedToolServer:
    """One live taskset-scoped (shared) server, as the rollouts see it: its eval-level
    `url` plus whether its runtime is `local` (host-reachable) — a remote one is an
    interception consumer, so the interception must be exposed for it to reach the
    `/state` channel (see `Env._requires_tunnel`). `runtime` is retained for translating
    that channel into the server's network. An `external` server (a
    config-`url` endpoint) was not launched by the framework and sits outside its state
    machinery entirely: rollouts get its URL bare — no state tag (and no per-rollout
    secret sent to a third party)."""

    url: str
    local: bool
    external: bool = False
    runtime: Runtime | None = field(default=None, repr=False)
    state_secret: str = field(default="", repr=False)


@contextlib.asynccontextmanager
async def serve_shared(toolsets: list[Toolset], harness_is_local: bool = True):
    """Start the taskset-scoped (shared) tool servers ONCE for a whole eval, each in its OWN
    `runtime`, and yield `{name: SharedToolServer}` reachable by every rollout's harness.
    Reachability mirrors a per-rollout tool, but there's no single harness runtime to read
    locality off — the caller (`Env.shared_tools`) passes the harness runtime's
    `harness_is_local`, so a host tool gets one host bridge (tunnel) when the harness runs
    remotely, and a remote tool runtime publishes its own URL. Torn down when the eval ends.
    A shared server is task-agnostic — the taskset carries no per-row data — so its `setup`
    gets no task (its `setup_task` is never called; the per-rollout servers fetch
    theirs over the `/task` channel)."""
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
            if cfg.url:  # already running remotely; nothing launched, nothing to bridge
                servers[name] = SharedToolServer(
                    url=cfg.url, local=False, external=True
                )
            else:
                state_secret = (
                    secrets.token_urlsafe(24) if toolset._state_cls is not State else ""
                )
                served = await stack.enter_async_context(
                    _serve(
                        toolset,
                        harness_is_local=harness_is_local,
                        state_secret=state_secret,
                    )
                )
                servers[name] = SharedToolServer(
                    url=served.url,
                    local=served.runtime.is_local,
                    runtime=served.runtime,
                    state_secret=state_secret,
                )
            logger.info("shared tool server '%s': %s", name, servers[name].url)
        yield servers


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
    if server.runtime is not None:
        state_url = server.runtime.host_url(state_url)
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
):
    """Bring up a rollout's tool servers and yield `{name: url}` the harness reaches: the
    task-scoped `toolsets` are launched by `serve` (placement off each one's `config`; the
    server fetches its task over the interception `/task` channel), and the
    taskset-scoped `shared` servers — already
    running eval-level (see `serve_shared`) — join under their per-rollout state tag.
    `state_secret` is private to task-scoped servers; shared servers keep an
    eval-level service secret and receive only signed `state_route` coordinates.
    `state_base` is universally reachable from either placement."""
    urls: dict[str, str] = {}
    async with contextlib.AsyncExitStack() as stack:
        for name, server in (shared or {}).items():
            if server.external:
                # Not ours: a pre-existing endpoint with no vf state channel. Pass the URL
                # through bare — a state tag would be useless, and the per-rollout secret
                # must not ride the query string to a third-party host.
                urls[name] = harness_runtime.host_url(server.url)
                logger.info("tool server '%s' (shared, external): %s", name, server.url)
                continue
            url = harness_runtime.host_url(server.url) if server.local else server.url
            urls[name] = _shared_url_for_rollout(server, url, state_base, state_route)
            logger.info("tool server '%s' (shared): %s", name, server.url)
        for toolset in toolsets:
            name = toolset.server_name
            if name in urls:
                raise ToolsetError(
                    f"tool server name '{name}' is declared both taskset-scoped (shared) "
                    f"and task-scoped — pick one scope, or give one a distinct TOOL_PREFIX"
                )
            cfg = toolset.config
            if cfg.url:
                urls[name] = harness_runtime.host_url(cfg.url)
                logger.info("tool server '%s' (remote): %s", name, cfg.url)
            else:
                urls[name] = await stack.enter_async_context(
                    serve(
                        toolset,
                        harness_runtime,
                        state_secret=state_secret,
                        state_base=state_base,
                    )
                )
                logger.info("tool server '%s': %s", name, urls[name])
        yield urls
