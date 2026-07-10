from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import shlex
import sys
import tarfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from verifiers.v1.errors import RolloutError, ToolsetError, UserError
from verifiers.v1.mcp.server import STATE_SECRET_PARAM, STATE_URL_PARAM, ServerBase
from verifiers.v1.runtimes import (
    HOST,
    Runtime,
    make_runtime,
    reachable_url,
    runtime_is_local,
)
from verifiers.v1.runtimes.base import _ENSURE_UV
from verifiers.v1.types import Messages

if TYPE_CHECKING:
    from verifiers.v1.mcp.toolset import Toolset
    from verifiers.v1.mcp.user import User

logger = logging.getLogger(__name__)

# Sandboxed servers install the working tree, so only wheel inputs need to cross the boundary.
VF_BUILD_INPUTS = ("pyproject.toml", "README.md", "LICENSE", "verifiers")

# A user ends the trajectory through shared state and `@stop`, not through this return value.
Respond = Callable[[str], Awaitable[Messages]]

# A server can pass its in-runtime probe before its host-facing connection settles.
_USER_CONNECT_ATTEMPTS = 12
_USER_CONNECT_BACKOFF = 0.2  # seconds, exponential up to the cap
_USER_CONNECT_MAX_BACKOFF = 2.0

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
    setup = (
        f"{_ENSURE_UV}; set -e; "
        f'for t in {root}/*.tar.gz; do tar -xzf "$t" -C {root}; done && '
        f"uv venv {venv} && "
        f"uv pip install --python {venv} {root}/{shlex.quote(vf.name)} && "
        f"uv pip install --python {venv} {root}/{shlex.quote(env.name)}"
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
    """Poll the server's port file without stacking the runtime's own read retries."""
    reader = getattr(runtime, "inner", runtime)
    for _ in range(180):
        with contextlib.suppress(Exception):
            data = (await reader.read(path)).decode().strip()
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
    if runtime.type == "subprocess":
        python = sys.executable
    else:
        python = await _install_in_sandbox(server, runtime)
    log = f"vf_tool_{server.server_name}.log"
    await runtime.run_background([python, "-m", type(server).__module__], env, log)
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
async def serve(
    server: ServerBase,
    harness_runtime: Runtime | None = None,
    for_host: bool = False,
    harness_is_local: bool = True,
    *,
    state_port: int | None = None,
    state_secret: str = "",
    state_base: str | None = None,
):
    cfg = server.config
    async with contextlib.AsyncExitStack() as stack:
        if getattr(cfg, "colocated", False) and harness_runtime is not None:
            runtime = harness_runtime
        else:
            runtime = make_runtime(cfg.runtime)
            await runtime.start()
            stack.push_async_callback(runtime.stop)
        # Only consumers outside the server runtime need its fixed published port. Colocated tools
        # use independent OS-assigned ports, avoiding clashes on the runtime's service port.
        exposed = for_host or runtime is not harness_runtime
        state_url = None
        if state_port is not None:
            # Colocated servers reuse the harness's interception URL. A server in another runtime
            # needs its own bridge to the host state port.
            if state_base is not None and runtime is harness_runtime:
                base = state_base
            else:
                base = await stack.enter_async_context(
                    reachable_url(HOST, state_port, consumer=runtime)
                )
            state_url = f"{base.rstrip('/')}/state"
        port = await serve_in_runtime(
            server,
            runtime,
            exposed=exposed,
            state_url=state_url,
            state_secret=state_secret,
        )
        # User simulators are host-driven; tool servers are harness-facing.
        consumer = HOST if for_host else harness_runtime
        base = await stack.enter_async_context(
            reachable_url(
                runtime, port, consumer=consumer, consumer_is_local=harness_is_local
            )
        )
        yield f"{base.rstrip('/')}/mcp"


@dataclass(frozen=True)
class SharedToolServer:
    url: str
    local: bool


@contextlib.asynccontextmanager
async def serve_shared(toolsets: list[Toolset], harness_is_local: bool = True):
    """Serve one taskset-scoped tool instance for an environment worker."""
    servers: dict[str, SharedToolServer] = {}
    async with contextlib.AsyncExitStack() as stack:
        for toolset in toolsets:
            cfg = toolset.config
            name = toolset.server_name
            if cfg.url:
                servers[name] = SharedToolServer(url=cfg.url, local=False)
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
    state_port: int | None = None,
    state_secret: str = "",
    state_base: str | None = None,
):
    """Serve rollout tools; task and state reach servers through the state channel."""
    urls: dict[str, str] = {}
    async with contextlib.AsyncExitStack() as stack:
        for name, server in (shared or {}).items():
            tool_state_base = state_base
            # A remote shared server cannot reach a local harness's loopback state URL.
            if state_base and harness_runtime.is_local and not server.local:
                tool_state_base = await stack.enter_async_context(
                    reachable_url(HOST, state_port, consumer_is_local=False)
                )
            urls[name] = _shared_url_for_rollout(
                server.url, tool_state_base, state_secret
            )
            # The tagged URL contains the bearer secret; log only the untagged base URL.
            logger.info("tool server '%s' (shared): %s", name, server.url)
        for toolset in toolsets:
            name = toolset.server_name
            cfg = toolset.config
            if cfg.url:
                urls[name] = cfg.url
                logger.info("tool server '%s' (remote): %s", name, cfg.url)
            else:
                urls[name] = await stack.enter_async_context(
                    serve(
                        toolset,
                        harness_runtime,
                        state_port=state_port,
                        state_secret=state_secret,
                        state_base=state_base,
                    )
                )
                logger.info("tool server '%s': %s", name, urls[name])
        yield urls


@contextlib.asynccontextmanager
async def connect_user(url: str) -> AsyncIterator[Respond]:
    """Connect to a user server, retrying only initial connection failures.

    Body errors propagate unchanged; transport failures after connecting become `UserError`. The
    session stays in this frame so AnyIO cancellation scopes remain correctly nested.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from verifiers.v1.dialects import parse_message

    last_exc: Exception | None = None
    for attempt in range(_USER_CONNECT_ATTEMPTS):
        connected = in_body = False
        try:
            async with (
                streamable_http_client(url) as (read, write, *_),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                connected = True

                async def respond(message: str) -> Messages:
                    result = await session.call_tool("respond", {"message": message})
                    texts = [
                        b.text
                        for b in result.content
                        if getattr(b, "type", None) == "text"
                    ]
                    data = json.loads("\n".join(texts))
                    return [parse_message(m) for m in data["messages"]]

                # Errors thrown into the yield belong to the harness body, not the connection.
                in_body = True
                yield respond
                in_body = False
            return
        except RolloutError:
            raise
        except Exception as e:
            if in_body:
                raise
            if connected:
                # Raw transport groups bypass rollout handling, so attribute the loss here.
                raise UserError(f"user server at {url} connection lost: {e!r}") from e
            last_exc = e
            await asyncio.sleep(
                min(_USER_CONNECT_BACKOFF * 2**attempt, _USER_CONNECT_MAX_BACKOFF)
            )
    raise UserError(
        f"user server at {url} unreachable after {_USER_CONNECT_ATTEMPTS} attempts: {last_exc!r}"
    )


@contextlib.asynccontextmanager
async def serve_user(
    user: User | None,
    harness_runtime: Runtime | None = None,
    *,
    state_port: int | None = None,
    state_secret: str = "",
    state_base: str | None = None,
) -> AsyncIterator[Respond | None]:
    """Serve a rollout user; task and state reach it through the state channel."""
    if user is None:
        yield None
        return
    async with serve(
        user,
        harness_runtime,
        for_host=True,
        state_port=state_port,
        state_secret=state_secret,
        state_base=state_base,
    ) as url:
        async with connect_user(url) as respond:
            yield respond
