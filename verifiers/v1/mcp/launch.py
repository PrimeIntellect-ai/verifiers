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
import random
import shlex
import socket
import sys
import tarfile
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from verifiers.v1.errors import ProgramError, RolloutError
from verifiers.v1.mcp.server import ServerBase
from verifiers.v1.runtimes import Runtime, host_endpoint, make_runtime
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

# The model's last assistant text in; the next user messages + a done flag out.
Respond = Callable[[str], Awaitable[tuple[Messages, bool]]]

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


@contextlib.asynccontextmanager
async def connect_user(url: str) -> AsyncIterator[Respond]:
    """Open an MCP client session to a user server at `url` and yield an async
    `respond(message)` that calls its `respond` tool, parsing the JSON it returns
    (`{"messages": [...], "done": bool}`) into typed `(messages, done)`.

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
        connected = False
        try:
            async with (
                streamable_http_client(url) as (read, write, *_),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                connected = True

                async def respond(message: str) -> tuple[Messages, bool]:
                    result = await session.call_tool("respond", {"message": message})
                    texts = [
                        b.text
                        for b in result.content
                        if getattr(b, "type", None) == "text"
                    ]
                    data = json.loads("\n".join(texts))
                    messages = [parse_message(m) for m in data["messages"]]
                    return messages, bool(data["done"])

                yield respond
            return
        except RolloutError:
            raise  # a real rollout error surfaced after connecting: propagate as-is
        except Exception as e:
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
) -> AsyncIterator[Respond | None]:
    """Bring a rollout's user server up (via the shared `serve` launcher, `for_host=True` since
    the framework drives the user from the HOST) and yield the async `respond` the interception
    server drives — or `None` when the taskset has no user server. Placement is the user's
    `config` (colocated in the agent's runtime, or its own); the rollout's `task` is shipped to
    the server for its `setup`."""
    if user is None:
        yield None
        return
    async with serve(user, task, agent_runtime, for_host=True) as url:
        async with connect_user(url) as respond:
            yield respond
