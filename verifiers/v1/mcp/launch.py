from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import io
import logging
import shlex
import sys
import tarfile
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from verifiers.v1.errors import ToolsetError
from verifiers.v1.interception.tunnel import PrimeTunnel
from verifiers.v1.mcp.server import STATE_SECRET_PARAM, STATE_URL_PARAM, ServerBase
from verifiers.v1.runtimes import (
    DockerConfig,
    Runtime,
    make_runtime,
    runtime_is_local,
)
from verifiers.v1.runtimes.base import _ENSURE_UV

if TYPE_CHECKING:
    from verifiers.v1.mcp.toolset import Toolset

logger = logging.getLogger(__name__)

# Sandboxed servers install the working tree, so only wheel inputs need to cross the boundary.
VF_BUILD_INPUTS = ("pyproject.toml", "README.md", "LICENSE", "verifiers")

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


# These can be gigabytes and are not package source, so uploading them can stall startup.
_TAR_EXCLUDE = {
    "__pycache__",
    ".venv",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}


@cache
def _tar_source(src: Path, members: tuple[str, ...] = ()) -> bytes:
    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return None if _TAR_EXCLUDE & set(info.name.split("/")) else info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for member in members or ("",):
            path = src / member if member else src
            if path.exists():
                tar.add(path, arcname=f"{src.name}/{member}".rstrip("/"), filter=keep)
    return buf.getvalue()


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
    root = "/tmp/vf-src"
    vf, env = _verifiers_root(), Path(source_dir)
    await runtime.write(f"{root}/{vf.name}.tar.gz", _tar_source(vf, VF_BUILD_INPUTS))
    await runtime.write(f"{root}/{env.name}.tar.gz", _tar_source(env))
    venv = "/tmp/vf-venv"
    # The upload carries no .git, so hatch-vcs falls back to version 0.0.0 — an env
    # package's `verifiers>=...` floor would then resolve PyPI verifiers OVER the local
    # build, silently running the server against a released (older) API. Pretend the
    # local version so the floor is satisfied by the build we uploaded.
    vf_version = importlib.metadata.version("verifiers")
    extras = ",".join(type(server).EXTRAS)
    setup = (
        f"{_ENSURE_UV}; set -e; "
        f'for t in {root}/*.tar.gz; do tar -xzf "$t" -C {root}; done && '
        f"uv venv {venv} && "
        f"SETUPTOOLS_SCM_PRETEND_VERSION={shlex.quote(vf_version)} "
        f"uv pip install --python {venv} {root}/{shlex.quote(vf.name)} && "
        f"uv pip install --python {venv} "
        f"{shlex.quote(f'{root}/{env.name}' + (f'[{extras}]' if extras else ''))}"
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

    Exposed remote servers must use the runtime's forwarded port. Local or colocated servers let
    the OS choose and report the result through a file. With a state channel, the server fetches
    the current rollout task from the adjacent `/task` endpoint rather than a launch argument.
    """
    env = {"VF_CONFIG": server.config.model_dump_json()}
    if runtime.type == "subprocess":
        # Keep provider temp files in the runtime workdir so cleanup removes them.
        assert runtime.info.id is not None
        env["TMPDIR"] = runtime.info.id
    if state_url:
        env["VF_STATE_URL"] = state_url
        env["VF_STATE_SECRET"] = state_secret
    if runtime.published_port is not None:
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
        ["python3", "-c", _PROBE, f"http://127.0.0.1:{port}/mcp"], {}
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
    - the server runs in a remote sandbox -> its own published URL (`expose`), reachable anywhere;
    - else it's host-local -> localhost to a local consumer, a host tunnel to a remote one."""
    if colocated:
        yield f"http://127.0.0.1:{port}"
    elif not service.is_local:  # in a remote sandbox → it publishes its own port
        yield await service.expose(port)
    elif consumer_is_local:  # local consumer → localhost, no public tunnel
        yield f"http://127.0.0.1:{port}"
    else:  # remote consumer → a host tunnel publishes the port outward
        async with PrimeTunnel().expose(port) as url:
            yield url


@contextlib.asynccontextmanager
async def serve(
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
        # Docker server has neither that lifecycle nor a published port after isolation;
        # reject it instead of silently leaving its requested policy unenforced.
        if (
            isinstance(cfg.runtime, DockerConfig)
            and cfg.runtime.network_isolated
            and not (colocated and harness_runtime is not None)
        ):
            raise ToolsetError(
                "Docker network policies are supported on the harness runtime; "
                f"server {server.server_name!r} must be colocated or use an "
                "unrestricted Docker runtime"
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
        if colocated and harness_runtime is not None and runtime.network_isolated:
            base = base.replace("127.0.0.1", "localhost", 1)
        elif not colocated and harness_runtime is not None:
            base = harness_runtime.host_url(base)
        yield f"{base.rstrip('/')}/mcp"


@dataclass(frozen=True)
class SharedToolServer:
    """One live taskset-scoped (shared) server, as the rollouts see it: its eval-level
    `url` plus whether its runtime is `local` (host-reachable) — a remote one is an
    interception consumer, so the interception must be exposed for it to reach the
    `/state` channel (see `Env._requires_tunnel`). An `external` server (a
    config-`url` endpoint) was not launched by the framework and sits outside its state
    machinery entirely: rollouts get its URL bare — no state tag (and no per-rollout
    secret sent to a third party)."""

    url: str
    local: bool
    external: bool = False


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
                    f"duplicate shared tool server name '{name}' in Taskset.tools — "
                    f"give one a distinct TOOL_PREFIX"
                )
            if type(toolset).setup_task is not ServerBase.setup_task:
                logger.warning(
                    "shared server %r overrides `setup_task`, but `setup_task` is NEVER "
                    "called for a taskset-scoped server (it's built once, task-agnostic) — "
                    "its per-task logic will not run. Move task-agnostic work into `setup`, "
                    "or declare it on `Task.tools` to run it per-rollout.",
                    name,
                )
            if cfg.url:  # already running remotely; nothing launched, nothing to bridge
                servers[name] = SharedToolServer(
                    url=cfg.url, local=False, external=True
                )
            else:
                url = await stack.enter_async_context(
                    serve(toolset, harness_is_local=harness_is_local)
                )
                servers[name] = SharedToolServer(
                    url=url, local=runtime_is_local(cfg.runtime)
                )
            logger.info("shared tool server '%s': %s", name, servers[name].url)
        yield servers


def _shared_url_for_rollout(url: str, state_base: str | None, state_secret: str) -> str:
    """Attach one rollout's state bridge to a shared server URL."""
    if not state_base:
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query[STATE_URL_PARAM] = f"{state_base.rstrip('/')}/state"
    query[STATE_SECRET_PARAM] = state_secret
    return urlunsplit(parts._replace(query=urlencode(query)))


@contextlib.asynccontextmanager
async def serve_tools(
    toolsets: list[Toolset],
    harness_runtime: Runtime,
    shared: dict[str, SharedToolServer] | None = None,
    *,
    state_secret: str = "",
    state_base: str | None = None,
):
    """Bring up a rollout's tool servers and yield `{name: url}` the harness reaches: the
    task-scoped `toolsets` are launched by `serve` (placement off each one's `config`; the
    server fetches its task over the interception `/task` channel), and the
    taskset-scoped `shared` servers — already
    running eval-level (see `serve_shared`) — join under their per-rollout state tag.
    `state_base`/`state_secret` wire each server to the interception server's shared-state
    channel — `state_base` is universally reachable, so every server (per-rollout or
    `shared`, any runtime) uses it directly."""
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
            urls[name] = _shared_url_for_rollout(url, state_base, state_secret)
            # The tagged URL contains the bearer secret; log only the untagged base URL.
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
