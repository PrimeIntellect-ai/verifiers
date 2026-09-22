from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import secrets
import shlex
import sys
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
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
    provision_runtime,
)
from verifiers.v1.runtimes.base import _ENSURE_UV
from verifiers.v1.state import State
from verifiers.v1.utils.aio import run_shielded

if TYPE_CHECKING:
    from verifiers.v1.mcp.toolset import Toolset

logger = logging.getLogger(__name__)

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


async def _install_packages(
    runtime: Runtime,
    key: str,
    requirements: list[str],
    uploads: dict[str, bytes],
    root: str,
) -> str:
    """Install one complete environment under the runtime lock, then publish it."""
    for path, data in uploads.items():
        await runtime.write(path, data)
    venv, temp, cache = (f"{root}/envs/{key}", f"{root}/tmp", f"{root}/cache")
    venv_q, temp_q, cache_q = map(shlex.quote, (venv, temp, cache))
    command = (
        f"set -e; mkdir -p {temp_q} {cache_q}; "
        f"export TMPDIR={temp_q} UV_CACHE_DIR={cache_q}; "
        f"{_ENSURE_UV}; uv venv --allow-existing {venv_q}; "
        f"uv pip install --python {venv_q} -- {shlex.join(requirements)}"
    )
    result = await runtime.run(["sh", "-c", command], {})
    if result.exit_code:
        raise ToolsetError(
            f"MCP package installation failed: {(result.stderr or result.stdout)[-2000:]}"
        )
    python = f"{venv}/bin/python"
    runtime._mcp_environments[key] = python
    return python


async def prepare_packages(packages: Sequence[str], runtime: Runtime) -> str:
    """Resolve all native servers' packages together before launching any of them."""
    if runtime.type == "subprocess":
        return sys.executable
    if not packages:
        return "python3"  # The image supplies the complete server environment.
    # Build scratch space and uv's cache belong on disk, not a VM's small /tmp tmpfs.
    root = str(PurePosixPath(runtime.config.workdir) / ".vf-mcp")
    uploads: dict[str, bytes] = {}
    requirements = []
    for requirement in sorted(set(packages)):
        path = Path(requirement)
        if path.is_absolute():
            if not path.name.endswith((".whl", ".tar.gz")):
                raise ToolsetError(
                    "build local MCP packages with uv build --sdist first"
                )
            data = await asyncio.to_thread(path.read_bytes)
            digest = hashlib.sha256(data).hexdigest()
            requirement = f"{root}/artifacts/{digest}/{path.name}"
            uploads[requirement] = data
        requirements.append(requirement)
    requirements = sorted(set(requirements))
    # Borrowers may change uv indexes or build settings through their process env.
    # Include that input too; never mutate an environment already used by a server.
    key = hashlib.sha256(
        json.dumps([requirements, runtime.env], sort_keys=True).encode()
    ).hexdigest()
    async with runtime._mcp_install_lock:
        if key not in runtime._mcp_environments:
            # Drain uploads, installation, and publication before releasing the lock,
            # including when the caller is cancelled more than once.
            await run_shielded(
                _install_packages(runtime, key, requirements, uploads, root)
            )
        return runtime._mcp_environments[key]


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
    python: str,
    state_url: str | None = None,
    state_secret: str = "",
) -> int:
    """Start a server and return its bound port.

    Exposed remote servers must use the runtime's forwarded port. Local or colocated servers let
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
    if exposed and runtime.published_port is not None:
        env["MCP_HOST"] = "0.0.0.0"
    fixed = runtime.published_port if exposed else None
    port_file = None
    if fixed is not None:
        env["MCP_PORT"] = str(fixed)
    else:
        port_file = f"/tmp/vf-port-{uuid.uuid4().hex}"
        env["MCP_PORT_FILE"] = port_file
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
    - else the runtime publishes the port (`expose`): a remote sandbox's URL is reachable
      anywhere, a host-local URL directly by a local consumer and through a host tunnel
      by a remote one."""
    if colocated:
        yield f"http://127.0.0.1:{port}"
        return
    url = await service.expose(port)
    if service.is_local and not consumer_is_local:
        async with PrimeTunnel().expose(urlsplit(url).port or 80) as public:
            yield public
    else:
        yield url


@dataclass(frozen=True)
class _ServedServer:
    url: str
    runtime: Runtime


@contextlib.asynccontextmanager
async def serve(
    server: ServerBase,
    harness_runtime: Runtime | None = None,
    harness_is_local: bool = True,
    *,
    state_secret: str = "",
    state_base: str | None = None,
    packages: Sequence[str] = (),
):
    """Serve one MCP server and yield its consumer-visible URL and runtime."""
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
            runtime = await stack.enter_async_context(provision_runtime(cfg.runtime))
        # Only consumers outside the server runtime need its fixed published port. Colocated tools
        # use independent OS-assigned ports, avoiding clashes on the runtime's service port.
        exposed = runtime is not harness_runtime
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
            exposed=exposed,
            state_url=state_url,
            state_secret=state_secret,
            python=await prepare_packages(packages, runtime),
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
async def serve_shared(
    toolsets: list[Toolset],
    harness_is_local: bool = True,
    *,
    packages: Sequence[str] = (),
):
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
                    serve(
                        toolset,
                        harness_is_local=harness_is_local,
                        state_secret=state_secret,
                        packages=packages,
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
    packages: Sequence[str] = (),
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
                served = await stack.enter_async_context(
                    serve(
                        toolset,
                        harness_runtime,
                        state_secret=state_secret,
                        state_base=state_base,
                        packages=packages,
                    )
                )
                urls[name] = served.url
                logger.info("tool server '%s': %s", name, urls[name])
        yield urls
