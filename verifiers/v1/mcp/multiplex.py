"""Fork-per-rollout isolation for a SHARED tool server.

A `shared` server is one process for the whole eval; `self.state` isolates its per-rollout state over
the interception channel (see `server`), but state that DOESN'T live in `self.state` — module globals,
a mutated in-memory object (an index the tool edits), relative-path on-disk writes — is still shared
across rollouts. `ToolsetConfig(fork=True)` isolates those too: the expensive `setup` runs ONCE in a
parent process, then a child is forked per rollout on first contact — the child inherits the warm
state copy-on-write (in-memory state isolated on write) and runs in a private working directory
(relative-path writes isolated). The parent is a thin async reverse proxy that pins each rollout to
its child and streams MCP traffic (so SSE works). So an ordinary stateful server is isolated per
rollout with no rollout-aware code.

The rollout key is the per-rollout secret the framework tags onto a shared server's URL
(`STATE_SECRET_PARAM`, see `serve_tools`) — which is only added for a local shared runtime, so fork
requires one. Children are reaped on a `POST /vf/close?<key>` from rollout teardown, by an idle TTL,
and when the parent exits (each child also dies with its parent via `PR_SET_PDEATHSIG`).

Caveats: Linux/fork only; do NOT use with CUDA/GPU state or background threads in the server (fork
copies neither safely) — the fork here is from a single-threaded loop, before any such state. Writes
to absolute paths outside the private CWD are not isolated.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import socket
import tempfile
import time
from urllib.parse import parse_qs

from verifiers.v1.mcp.server import STATE_SECRET_PARAM, _die_with_parent

logger = logging.getLogger(__name__)

CLOSE_PATH = "/vf/close"
"""The framework POSTs here (with the rollout key) on rollout teardown to reap that child promptly."""

_IDLE_TTL = float(
    os.environ.get("VF_FORK_TTL", "900")
)  # reap idle children after N seconds
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "content-length",
    "host",
}


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _Child:
    def __init__(self, pid: int, port: int, cwd: str) -> None:
        self.pid, self.port, self.cwd = pid, port, cwd
        self.last = time.monotonic()


def _serve_child(app, port: int, cwd: str) -> None:
    """In the forked child: die with the parent, move to a private CWD, and serve `app` on `port`
    with a fresh event loop (the inherited parent loop is abandoned). Never returns."""
    import uvicorn

    _die_with_parent()  # SIGKILL this child when the parent multiplexer dies (cleared by fork)
    os.chdir(cwd)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=port,
                    log_level="critical",
                    # exit promptly on SIGTERM at teardown — don't hang waiting on the long-lived
                    # MCP SSE connection to close (the parent SIGKILLs us / PR_SET_PDEATHSIG fires).
                    timeout_graceful_shutdown=0,
                )
            ).serve()
        )
    finally:
        os._exit(0)


async def _wait_up(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise RuntimeError(f"fork child on port {port} did not come up")


def serve_forked(app, sock: socket.socket) -> None:
    """Serve `app` on the bound `sock` as a fork-per-rollout multiplexer (see module docstring).
    `app` is the warm MCP ASGI app, built once in the parent after `setup`; forked children inherit
    it copy-on-write. `sock` is the socket the launcher reads the port back from."""
    import httpx
    import uvicorn

    base = tempfile.mkdtemp(prefix="vf_fork_")
    children: dict[str, _Child] = {}
    forking = asyncio.Lock()
    client = httpx.AsyncClient(timeout=None)

    async def ensure(key: str) -> _Child:
        async with forking:  # one fork at a time keeps the single-threaded fork clean
            child = children.get(key)
            if child is None:
                port = _free_port()
                cwd = os.path.join(base, key or "_default")
                os.makedirs(cwd, exist_ok=True)
                pid = os.fork()
                if pid == 0:
                    _serve_child(app, port, cwd)  # child: never returns
                children[key] = child = _Child(pid, port, cwd)
                await _wait_up(port)
                logger.info(
                    "fork: child pid=%d for rollout %s", pid, (key or "_default")[:12]
                )
            child.last = time.monotonic()
            return child

    async def reap(key: str) -> None:
        child = children.pop(key, None)
        if not child:
            return
        with contextlib.suppress(Exception):
            os.kill(child.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            os.waitpid(child.pid, 0)
        shutil.rmtree(child.cwd, ignore_errors=True)
        logger.info(
            "fork: reaped child pid=%d for rollout %s",
            child.pid,
            (key or "_default")[:12],
        )

    async def _respond(send, status: int, body: bytes) -> None:
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": body})

    async def proxy(scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            # Drive the proxy app's own (empty) lifespan; each child runs the real app's lifespan.
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        key = (
            parse_qs(scope.get("query_string", b"").decode()).get(STATE_SECRET_PARAM)
            or [""]
        )[0]
        if scope["path"] == CLOSE_PATH:
            await reap(key)
            await _respond(send, 200, b"closed")
            return
        body, more = b"", True
        while more:  # read the request body, then forward to the rollout's child
            msg = await receive()
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
        child = await ensure(key)
        headers = [
            (k.decode(), v.decode())
            for k, v in scope["headers"]
            if k.decode().lower() not in _HOP_BY_HOP
        ]
        qs = scope.get("query_string", b"").decode()
        url = f"http://127.0.0.1:{child.port}{scope['path']}" + (f"?{qs}" if qs else "")
        req = client.build_request(scope["method"], url, headers=headers, content=body)
        resp = await client.send(req, stream=True)
        out = [
            (k.encode(), v.encode())
            for k, v in resp.headers.items()
            if k.lower() not in _HOP_BY_HOP
        ]
        await send(
            {"type": "http.response.start", "status": resp.status_code, "headers": out}
        )
        async for chunk in resp.aiter_raw():
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        await resp.aclose()

    async def _reaper() -> None:
        while True:
            await asyncio.sleep(30)
            now = time.monotonic()
            for key in [k for k, c in children.items() if now - c.last > _IDLE_TTL]:
                await reap(key)

    async def _serve() -> None:
        asyncio.ensure_future(_reaper())
        # timeout_graceful_shutdown=0: exit promptly on SIGTERM (the runtime's teardown) instead of
        # hanging on the long-lived proxied SSE — so `finally` below SIGKILLs the children.
        await uvicorn.Server(
            uvicorn.Config(proxy, log_level="critical", timeout_graceful_shutdown=0)
        ).serve(sockets=[sock])

    try:
        asyncio.run(_serve())
    finally:
        for key in list(children):
            with contextlib.suppress(Exception):
                os.kill(children[key].pid, signal.SIGKILL)
        shutil.rmtree(base, ignore_errors=True)
