"""Host-side launching: bring a vf-native server up in a runtime and reach it.

`serve` is the single launcher (any server, any placement); `serve_tools` / `serve_shared` /
`serve_user` are thin wrappers for a rollout's tools, the eval's shared tools, and the user sim.
`serve_in_runtime` runs the server's module (`python -m <module>`) in a runtime — host (ambient) or
sandbox (after `_install_in_sandbox` uploads + installs the working-tree `verifiers` source + the env
package). `connect_user` is the MCP client the framework drives the user sim through.
"""

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
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from verifiers.v1.errors import ProgramError, RolloutError
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

# The verifiers source tree's wheel-build inputs — uploaded into a sandbox so it installs the
# developer's working-tree verifiers (deps resolve from PyPI off the uploaded pyproject), with no
# publish or git pin to keep in sync.
VF_BUILD_INPUTS = ["pyproject.toml", "README.md", "LICENSE", "verifiers"]

# The model's last assistant text in; the next user messages out. The user sim ends the trajectory
# by setting a flag on the shared `self.state` that the taskset's `@vf.stop` checks, not via a return
# flag — so this hands back only messages.
Respond = Callable[[str], Awaitable[Messages]]

# The colocated user server is up once its in-runtime probe passes, but under high concurrency
# it can still momentarily refuse a host connection. Retry the connect before giving up so a
# transient refusal doesn't fail the rollout.
_USER_CONNECT_ATTEMPTS = 12
_USER_CONNECT_BACKOFF = 0.2  # seconds, exponential up to the cap
_USER_CONNECT_MAX_BACKOFF = 2.0

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


_TAR_EXCLUDE = {
    "__pycache__",
    ".venv",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}
"""Directory names never uploaded to a sandbox — build/VCS/cache trees that aren't package source
and can be gigabytes (a `.venv` alone is many GB), which would otherwise stall the upload."""


def _tar_source(src: Path, members: list[str] | None = None) -> bytes:
    """Gzipped tarball of a local package dir, rooted at `src.name/`, skipping `_TAR_EXCLUDE` dirs.
    `members` limits it to those top-level entries (the verifiers tree only needs its package +
    project files); otherwise the whole dir (a small env package)."""

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return None if _TAR_EXCLUDE & set(info.name.split("/")) else info

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


async def log_tail(runtime: Runtime, log: str, limit: int = 2000) -> str:
    """The tail of a program's `log` file in `runtime`, empty if it can't be read — for
    enriching an error with what the program (e.g. a tool server) wrote before it died."""
    with contextlib.suppress(Exception):
        return (await runtime.read(log)).decode(errors="replace").strip()[-limit:]
    return ""


async def _read_back_port(runtime: Runtime, path: str) -> int:
    """The port the server bound and wrote to `path` in its runtime. The server writes it the moment
    it binds (before setup/serving), so this resolves quickly; poll because it's a separate process."""
    for _ in range(180):
        with contextlib.suppress(Exception):
            data = (await runtime.read(path)).decode().strip()
            if data.isdigit():
                return int(data)
        await asyncio.sleep(1)
    raise ProgramError(f"server did not report its port at {path} in its runtime")


async def serve_in_runtime(
    server: ServerBase,
    task,
    runtime: Runtime,
    *,
    exposed: bool,
    state_url: str | None = None,
    state_secret: str = "",
) -> int:
    """Start `server` inside `runtime` (background, by running its env module — `python -m <module>`,
    whose `__main__` calls `ServerBase.run()`), wait until it serves, and return the port it bound.
    An `exposed` server (reached from outside its runtime) binds the runtime's fixed `published_port`
    (modal/prime forward only that); otherwise the server binds an OS-assigned free port in its own
    environment and reports it back (`MCP_PORT_FILE`) — so the launcher never probes for a free port.
    The `config` + this rollout's `task` cross as env JSON (a `shared` server passes `None`). On a
    host (`subprocess`) runtime it runs with the eval's own interpreter; in a sandbox the working-tree
    `verifiers` source + the env package are uploaded and installed first (`_install_in_sandbox`)."""
    env = {"VF_CONFIG": server.config.model_dump_json()}
    if task is not None:
        env["VF_TASK_CLS"] = f"{type(task).__module__}:{type(task).__qualname__}"
        env["VF_TASK"] = task.model_dump_json()
    if state_url:  # the shared-state back-channel to this rollout's interception server
        env["VF_STATE_URL"] = state_url
        env["VF_STATE_SECRET"] = state_secret
    if (
        runtime.published_port is not None
    ):  # a self-publishing runtime (modal/prime) forwards to
        env["MCP_HOST"] = "0.0.0.0"  # all interfaces, not just loopback
    fixed = runtime.published_port if exposed else None
    port_file = None
    if fixed is not None:
        env["MCP_PORT"] = str(fixed)
    else:  # bind an OS-assigned free port in the server's own environment, reported back here
        port_file = f"/tmp/vf-port-{uuid.uuid4().hex}"
        env["MCP_PORT_FILE"] = port_file
    if runtime.type == "subprocess":  # host: verifiers + env module already installed
        python = sys.executable
    else:  # sandbox: upload + install the verifiers source + the env package
        python = await _install_in_sandbox(server, runtime)
    log = f"vf_tool_{server.server_name}.log"
    await runtime.run_background([python, "-m", type(server).__module__], env, log)
    if fixed is not None:
        port = fixed
    else:
        try:
            port = await _read_back_port(runtime, port_file)
        except ProgramError as e:
            raise ProgramError(f"{e}: {await log_tail(runtime, log)}") from e
    probe = await runtime.run(
        ["python3", "-c", _PROBE, f"http://127.0.0.1:{port}/mcp"], {}
    )
    if probe.exit_code != 0:
        raise ProgramError(
            f"tool server {server.server_name!r} not serving in runtime: {await log_tail(runtime, log)}"
        )
    return port


@contextlib.asynccontextmanager
async def serve(
    server: ServerBase,
    task,
    agent_runtime: Runtime | None = None,
    for_host: bool = False,
    agent_is_local: bool = True,
    *,
    state_port: int | None = None,
    state_secret: str = "",
):
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
    if shared and getattr(cfg, "fork", False) and not runtime_is_local(cfg.runtime):
        raise ValueError(
            f"forked shared server {server.server_name!r} needs a local runtime — the per-rollout "
            "key is only tagged onto a local shared server's URL (set its runtime back to subprocess)"
        )
    async with contextlib.AsyncExitStack() as stack:
        if cfg.colocated and agent_runtime is not None:
            runtime = agent_runtime
        else:
            runtime = make_runtime(cfg.runtime)
            await runtime.start()
            stack.push_async_callback(runtime.stop)
        # `exposed` = the port is reached from OUTSIDE the server's runtime — a `for_host` server (the
        # host reaches it) or a tool in its own runtime (the harness, elsewhere, reaches it) — so it
        # binds the runtime's fixed published_port. A colocated tool is reached in-sandbox at localhost
        # and binds an OS-assigned free port instead (two colocated servers can't then clash on the
        # published SERVICE_PORT). `serve_in_runtime` returns the actual bound port.
        exposed = for_host or runtime is not agent_runtime
        # The shared-state channel: the interception server is a HOST service the server reaches from
        # its own runtime — localhost when local, a tunnel when remote. Shared/eval-level servers get
        # no channel (state is per-rollout; `state_port` is None for them).
        state_url = None
        if state_port is not None:
            state_base = await stack.enter_async_context(
                reachable_url(HOST, state_port, consumer=runtime)
            )
            state_url = f"{state_base.rstrip('/')}/state"
        port = await serve_in_runtime(
            server,
            task,
            runtime,
            exposed=exposed,
            state_url=state_url,
            state_secret=state_secret,
        )
        # Who consumes the server decides reachability (see `reachable_url`): a user sim is reached
        # by the host (`for_host`); a tool by the harness — its `agent_runtime` per rollout, or, for
        # a shared eval-level tool with no single agent, just the harness locality (`agent_is_local`).
        consumer = HOST if for_host else agent_runtime
        base = await stack.enter_async_context(
            reachable_url(
                runtime, port, consumer=consumer, consumer_is_local=agent_is_local
            )
        )
        yield f"{base.rstrip('/')}/mcp"


@contextlib.asynccontextmanager
async def serve_shared(toolsets: list[Toolset], agent_is_local: bool = True):
    """Start the SHARED tool servers (placement `shared`) ONCE for a whole eval, each in its OWN
    `runtime`, and yield `{name: url}` reachable by every rollout's harness. Reachability mirrors a
    per-rollout tool, but there's no single agent runtime to read locality off — the caller
    (`Environment.shared_tools`) passes the harness runtime's `agent_is_local`, so a host tool gets
    one host bridge (tunnel) when the harness runs remotely, and a remote tool runtime publishes its
    own URL. Torn down when the eval ends. A shared server is task-agnostic, so its `setup` gets no
    task (`serve(toolset, None)`)."""
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
                urls[name] = await stack.enter_async_context(
                    serve(toolset, None, agent_is_local=agent_is_local)
                )
            logger.info("shared tool server '%s': %s", name, urls[name])
        yield urls


def _shared_url_for_rollout(
    url: str, toolset: Toolset, state_port: int | None, state_secret: str
) -> str:
    """Tag a `shared` server's eval-level URL with this rollout's state-channel coordinates, so the
    one shared process serves each rollout its OWN `self.state` (the per-rollout isolation a writable
    shared server needs — read-only ones simply ignore it). Only when the shared server runs on a
    local runtime, which reaches the host interception server at localhost; a remote shared runtime
    gets the plain URL (no per-rollout state — read-only-safe, as before). The secret is the bearer
    token the harness already holds, so tagging it on the URL is no new exposure."""
    if state_port is None or not runtime_is_local(toolset.config.runtime):
        return url
    state_url = f"http://127.0.0.1:{state_port}/state"
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query[STATE_URL_PARAM] = state_url
    query[STATE_SECRET_PARAM] = state_secret
    return urlunsplit(parts._replace(query=urlencode(query)))


async def _reap_forked_child(shared_url: str, secret: str) -> None:
    """Best-effort: POST `/vf/close` to reap this rollout's forked child on teardown (a `fork` shared
    server, see `multiplex`); the multiplexer also reaps by idle TTL and when it exits."""
    close_url = (
        f"{shared_url.rsplit('/mcp', 1)[0]}/vf/close?{STATE_SECRET_PARAM}={secret}"
    )
    with contextlib.suppress(Exception):
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(close_url)


@contextlib.asynccontextmanager
async def serve_tools(
    toolsets: list[Toolset],
    agent_runtime: Runtime,
    task,
    shared_urls: dict[str, str] | None = None,
    *,
    state_port: int | None = None,
    state_secret: str = "",
):
    """Bring up a rollout's tool servers and yield `{name: url}` the harness reaches. A `shared`
    toolset reuses the eval-level instance in `shared_urls`; the rest are launched by `serve`
    (placement off each one's `config`, the rollout's `task` for its `setup`) — so different
    servers can run in different runtimes. `state_port`/`state_secret` wire each per-rollout server to
    the interception server's shared-state channel."""
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
                urls[name] = _shared_url_for_rollout(
                    shared_urls[name], toolset, state_port, state_secret
                )
                # log the untagged base, NOT urls[name] — the per-rollout tag carries the rollout's
                # bearer secret (`vf_state_secret`), which must not reach a log sink
                logger.info("tool server '%s' (shared): %s", name, shared_urls[name])
                if getattr(cfg, "fork", False) and state_secret:
                    # reap this rollout's forked child when the rollout's tool-serving scope exits
                    stack.push_async_callback(
                        _reap_forked_child, shared_urls[name], state_secret
                    )
            else:
                urls[name] = await stack.enter_async_context(
                    serve(
                        toolset,
                        task,
                        agent_runtime,
                        state_port=state_port,
                        state_secret=state_secret,
                    )
                )
                logger.info("tool server '%s': %s", name, urls[name])
        yield urls


@contextlib.asynccontextmanager
async def connect_user(url: str) -> AsyncIterator[Respond]:
    """Open an MCP client session to a user server at `url` and yield an async
    `respond(message)` that calls its `respond` tool, parsing the JSON it returns
    (`{"messages": [...]}`) into typed `Messages`. End-of-trajectory is signalled out-of-band: the
    server sets a flag on the shared state (a taskset `@vf.stop` checks it), not in this reply.

    Retries the connect — under high concurrency the colocated user server can be slow to
    accept (or briefly refuse) a connection. A server that stays unreachable raises
    `ProgramError` (a captured, retryable rollout error), so a transport failure never escapes
    as a raw `ExceptionGroup`/`ConnectError` that would bypass rollout error handling and crash
    the batch. The connect is entered and exited in this one frame so anyio's cancel scopes stay
    correctly nested."""
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

                in_body = True  # while the harness drives; an error here is the body's, not ours
                yield respond
                in_body = False
            return
        except RolloutError:
            raise  # a real rollout error surfaced after connecting: propagate as-is
        except Exception as e:
            if in_body:
                raise  # the harness body raised (thrown back at the yield): propagate untouched,
                # don't mislabel it as a connection loss
            if connected:
                # the user-sim connection broke mid-rollout/teardown (e.g. the colocated server
                # was killed under memory pressure): capture it as a retryable rollout error
                # instead of letting the raw transport ExceptionGroup escape and crash the batch
                raise ProgramError(
                    f"user server at {url} connection lost: {e!r}"
                ) from e
            last_exc = e  # the connect itself failed: back off and retry
            await asyncio.sleep(
                min(_USER_CONNECT_BACKOFF * 2**attempt, _USER_CONNECT_MAX_BACKOFF)
            )
    raise ProgramError(
        f"user server at {url} unreachable after {_USER_CONNECT_ATTEMPTS} attempts: {last_exc!r}"
    )


@contextlib.asynccontextmanager
async def serve_user(
    user: User | None,
    task,
    agent_runtime: Runtime | None = None,
    *,
    state_port: int | None = None,
    state_secret: str = "",
) -> AsyncIterator[Respond | None]:
    """Bring a rollout's user server up (via the shared `serve` launcher, `for_host=True` since
    the framework drives the user from the HOST) and yield the async `respond` the interception
    server drives — or `None` when the taskset has no user server. Placement is the user's
    `config` (colocated in the agent's runtime, or its own); the rollout's `task` is shipped to
    the server for its `setup`. `state_port`/`state_secret` wire it to the shared-state channel — how
    the user sim's `respond` reads/writes `self.state` (and ends the trajectory via a flag a taskset
    `@vf.stop` checks)."""
    if user is None:
        yield None
        return
    async with serve(
        user,
        task,
        agent_runtime,
        for_host=True,
        state_port=state_port,
        state_secret=state_secret,
    ) as url:
        async with connect_user(url) as respond:
            yield respond
