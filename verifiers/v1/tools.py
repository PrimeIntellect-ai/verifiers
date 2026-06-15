"""Tools: how a task gives the harness tools, via tool servers it declares.

A taskset returns `Tools`s from `Taskset.tools`. A server is either a
single-file uv script the harness runs (`script` — its only runtime dep is `uv`, which
resolves the script's PEP 723 inline deps, so it runs in any runtime: host, the harness's
runtime when colocated, or its own) or an already-running remote endpoint (`url`, e.g.
deepwiki). The server binds the port passed via `MCP_PORT`.

`serve_tools` brings a task's servers up for a rollout — colocated in the harness's runtime
(reached via localhost) or in their own `tools.runtime` (reached via that runtime's
`public_url`, or the harness runtime's `expose` for a tunnel) — and yields `{name: url}`;
`serve_shared` brings up shared ones once per eval. `run_mcp_server` is the server-side
launcher. The wire types the model sees (`Tool`, `ToolCall`, …) live in `types`.
"""

import contextlib
import contextvars
import logging
import os
import random
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, quote, urlsplit

from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes import Runtime, RuntimeConfig, make_runtime
from verifiers.v1.runtimes.base import _ENSURE_UV

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# The framework tags every tool/user-server URL it hands out with the rollout's id as this
# query param (in `serve_tools`); `run_mcp_server` reads it back into `_rollout_id` per
# request. A server reads it via `current_rollout_id()` to scope (multiplex) its state by
# rollout. Entirely framework-side — the harness/program carry the URL verbatim.
ROLLOUT_ID_PARAM = "rollout_id"
_rollout_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "vf_rollout_id", default=None
)


def current_rollout_id() -> str | None:
    """The id of the rollout that issued the in-flight tool call, or None — set per request
    by `run_mcp_server` from the framework-injected `rollout_id` URL param. Lets a shared
    tool server namespace per-rollout state so concurrent rollouts don't corrupt each other."""
    return _rollout_id.get()


def _with_rollout_id(url: str, rollout_id: str | None) -> str:
    """Tag a framework-built tool-server URL with the rollout id so the server can read it."""
    if not rollout_id:
        return url
    sep = "&" if urlsplit(url).query else "?"
    return f"{url}{sep}{ROLLOUT_ID_PARAM}={quote(rollout_id, safe='')}"


@dataclass(frozen=True)
class Tools:
    """A tool server exposing tools to the model (as `<name>_<tool>`). Set exactly one
    of `script` (a single-file uv script we run) or `url` (already-running remote)."""

    name: str
    script: bytes | None = None
    """A self-contained uv script (PEP 723 inline deps) that serves streamable HTTP on
    the port in `MCP_PORT` — e.g. via `run_mcp_server`. Its only runtime dep is `uv`."""
    command: list[str] = field(default_factory=list)
    """Alternative to `script`: a ready-to-run argv that serves on `MCP_PORT` (for a
    server whose deps are already present where it runs)."""
    env: dict[str, str] = field(default_factory=dict)
    """Extra env for the script/command process (e.g. per-task metadata)."""
    url: str | None = None
    """An already-running streamable-HTTP MCP endpoint the harness connects to."""
    headers: dict[str, str] = field(default_factory=dict)
    """HTTP headers sent to a remote `url` (e.g. auth)."""


def run_mcp_server(mcp: "FastMCP", multiplex: bool = False) -> None:
    """Serve a FastMCP server on the port the harness passes via `MCP_PORT`, mounting
    streamable HTTP at `/mcp`. Call this at the end of a uv-script tool server. Each request's
    `rollout_id` URL param is exposed to the server's tools via `current_rollout_id()`.

    With `multiplex=True`, serve as a fork-per-rollout multiplexer (see `verifiers.v1.multiplex`):
    expensive setup runs once in the parent, then each rollout gets its own forked child
    (copy-on-write memory + a private working dir) — so an ordinary *stateful* server is
    isolated per rollout automatically, with no `current_rollout_id()` namespacing in the
    tool code."""
    if multiplex:
        from verifiers.v1.multiplex import run_multiplexed

        run_multiplexed(mcp)
        return

    import uvicorn

    port = int(os.environ["MCP_PORT"])
    app = mcp.streamable_http_app()

    async def with_rollout_id(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        params = parse_qs(scope.get("query_string", b"").decode())
        token = _rollout_id.set((params.get(ROLLOUT_ID_PARAM) or [None])[0])
        try:
            await app(scope, receive, send)
        finally:
            _rollout_id.reset(token)

    config = uvicorn.Config(
        with_rollout_id, host="127.0.0.1", port=port, log_level="critical"
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


async def serve_in_runtime(server: Tools, runtime: Runtime, port: int) -> None:
    """Run `server` inside `runtime` on `port` (background) and wait until it serves.
    A `script` runs via uv (its deps come from the script); a `command` must already be
    runnable in the runtime."""
    log = f"vf_tool_{server.name}.log"
    if server.script:
        script = f"vf_tool_{server.name}.py"
        await runtime.write(script, server.script)
        argv = ["sh", "-c", f"{_ENSURE_UV}; exec uv run --quiet {script}"]
    else:
        argv = list(server.command)
    await runtime.run_background(argv, {**server.env, "MCP_PORT": str(port)}, log)
    probe = await runtime.run(
        ["python3", "-c", _PROBE, f"http://127.0.0.1:{port}/mcp"], {}
    )
    if probe.exit_code != 0:
        tail = ""
        with contextlib.suppress(Exception):
            tail = (await runtime.read(log)).decode(errors="replace").strip()[-2000:]
        raise ProgramError(
            f"tool server {server.name!r} not serving in runtime: {tail}"
        )


async def _resolve_url(tool_runtime: Runtime, agent_runtime: Runtime, port: int) -> str:
    """A URL the harness can reach the tool's `port` at: the tool runtime publishes it if
    it can (a prime sandbox), else the harness runtime bridges the host port (localhost or
    a tunnel)."""
    base = await tool_runtime.public_url(port) or await agent_runtime.expose(port)
    return f"{base.rstrip('/')}/mcp"


@contextlib.asynccontextmanager
async def serve_shared(tools: list[Tools], tool_runtime_config: RuntimeConfig):
    """Start shared tool servers ONCE for a whole eval (each in its own `tools.runtime`)
    and yield `{name: url}` reachable by every rollout's harness — a prime tool runtime
    publishes its port (works for any harness), a host one is reached at localhost (works
    for host-network harnesses). Torn down when the eval ends. Used by `Environment` so an
    expensive corpus is built once, not per rollout."""
    tool_runtimes: list[Runtime] = []
    urls: dict[str, str] = {}
    try:
        for server in tools:
            if server.url:
                urls[server.name] = server.url
                continue
            port = _free_port()
            tool_runtime = make_runtime(tool_runtime_config)
            tool_runtimes.append(tool_runtime)
            await tool_runtime.start()
            await serve_in_runtime(server, tool_runtime, port)
            base = await tool_runtime.public_url(port) or f"http://127.0.0.1:{port}"
            urls[server.name] = f"{base.rstrip('/')}/mcp"
            logger.info("shared tool server '%s': %s", server.name, urls[server.name])
        yield urls
    finally:
        for tool_runtime in tool_runtimes:
            with contextlib.suppress(Exception):
                await tool_runtime.stop()


@contextlib.asynccontextmanager
async def serve_tools(
    tools: list[Tools],
    agent_runtime: Runtime,
    colocated: bool = False,
    host_reachable: bool = False,
    tool_runtime_config: RuntimeConfig | None = None,
    shared_urls: dict[str, str] | None = None,
    rollout_id: str | None = None,
):
    """Bring up the declared tool servers for a rollout; yield `{name: url}`. A colocated
    server is reached in-sandbox (localhost) by default — set `host_reachable` for one the
    framework consumes from the host (a user simulator), so its port is published back to the
    host (a remote sandbox's `public_url`, else localhost). Each framework-built URL is tagged
    with `rollout_id` so a (shared) server can scope its state per rollout."""
    shared_urls = shared_urls or {}
    tool_runtimes: list[Runtime] = []  # per-rollout tool runtimes to tear down
    urls: dict[str, str] = {}
    try:
        for server in tools:
            if server.url:  # already running remotely
                urls[server.name] = server.url
                logger.info("tool server '%s' (remote): %s", server.name, server.url)
            elif server.name in shared_urls:  # one shared instance, started eval-level
                urls[server.name] = _with_rollout_id(
                    shared_urls[server.name], rollout_id
                )
                logger.info(
                    "tool server '%s' (shared): %s",
                    server.name,
                    shared_urls[server.name],
                )
            elif colocated:  # in the given runtime
                port = _free_port()
                await serve_in_runtime(server, agent_runtime, port)
                # The model reaches a colocated tool in-sandbox (localhost); a host-consumed
                # one (a user simulator) needs the port published back to the host.
                base = (
                    (await agent_runtime.public_url(port) or f"http://127.0.0.1:{port}")
                    if host_reachable
                    else f"http://127.0.0.1:{port}"
                )
                urls[server.name] = _with_rollout_id(
                    f"{base.rstrip('/')}/mcp", rollout_id
                )
                logger.info("tool server '%s' colocated on port %d", server.name, port)
            else:  # its own runtime; reachability resolved per where it runs
                port = _free_port()
                tool_runtime = make_runtime(tool_runtime_config)
                tool_runtimes.append(tool_runtime)
                await tool_runtime.start()
                await serve_in_runtime(server, tool_runtime, port)
                urls[server.name] = _with_rollout_id(
                    await _resolve_url(tool_runtime, agent_runtime, port), rollout_id
                )
                logger.info(
                    "tool server '%s' on %s: %s",
                    server.name,
                    tool_runtime_config.type,
                    urls[server.name],
                )
        yield urls
    finally:
        for tool_runtime in tool_runtimes:
            with contextlib.suppress(Exception):
                await tool_runtime.stop()
