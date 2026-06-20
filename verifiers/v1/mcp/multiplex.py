"""Fork-per-rollout isolation for a SHARED tool server.

A `shared` server is one process for the whole eval; `self.state` isolates its per-rollout state over
the interception channel (see `server`), but state that DOESN'T live in `self.state` — module globals,
a mutated in-memory object (an index the tool edits), relative-path on-disk writes — is still shared
across rollouts. `ToolsetConfig(fork=True)` isolates those too: the expensive `setup` runs ONCE in a
parent process, then a child is forked per rollout on first contact — the child inherits the warm
state copy-on-write (in-memory state isolated on write), runs in a private working directory
(relative-path writes isolated), and runs `setup_task` for its rollout's task (fetched from the
interception server's `/task` channel — a shared server gets no task via env). The parent is a thin
async reverse proxy that pins each rollout to its child and streams MCP traffic (so SSE works). So an
ordinary stateful per-rollout server (expensive `setup` + per-rollout `setup_task`, e.g.
`wikispeedia-v1`) is isolated per rollout with no rollout-aware code.

The rollout key is the per-rollout secret the framework tags onto a shared server's URL
(`STATE_SECRET_PARAM`, see `serve_tools`), alongside the reachable interception base
(`STATE_URL_PARAM`) for this rollout's `/state` + `/task`. The proxy routes by the key (intra-sandbox,
no host needed); the child reaches `/state` + `/task` over that base, which `serve_tools` makes
reachable from the shared server's runtime (localhost, or a host tunnel when it's remote) — so fork
works on any harness/runtime combo. A child lives for its whole rollout: it is reaped on a `POST
/vf/close?<key>` from rollout teardown, and on parent exit (each child also dies with the parent via
`PR_SET_PDEATHSIG`). There is no idle reaper — idle time can't tell a slow-but-live rollout from a
leaked one, and reaping a live rollout's child would re-run `setup_task` on the next call and wipe its
in-process state.

Caveats: Linux/fork only; do NOT use with CUDA/GPU state or background threads in the server (fork
copies neither safely) — the fork here is from a single-threaded loop, before any such state. Writes
to absolute paths outside the private CWD are not isolated.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import shutil
import signal
import socket
import tempfile
import time
from urllib.parse import parse_qs

from verifiers.v1.mcp.server import (
    STATE_SECRET_PARAM,
    STATE_URL_PARAM,
    _die_with_parent,
)

logger = logging.getLogger(__name__)

CLOSE_PATH = "/vf/close"
"""The framework POSTs here (with the rollout key) on rollout teardown to reap that child promptly."""

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "content-length",
    "host",
}


class _Child:
    def __init__(self, pid: int, port: int, cwd: str) -> None:
        self.pid, self.port, self.cwd = pid, port, cwd
        # set once the child is serving; reap() also sets it, to wake any waiter
        self.ready = asyncio.Event()


def _serve_child(
    app, sock: socket.socket, cwd: str, server, state_url: str, secret: str
) -> None:
    """In the forked child: die with the parent, move to a private CWD, run `setup_task` for this
    rollout's task (over the per-request state channel), and serve `app` on the inherited (already
    bound) `sock` with a fresh event loop (the inherited parent loop is abandoned). Never returns."""
    import uvicorn

    _die_with_parent()  # SIGKILL this child when the parent multiplexer dies (cleared by fork)
    os.chdir(cwd)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server._setup_task_from_channel(state_url, secret))
        loop.run_until_complete(
            uvicorn.Server(
                uvicorn.Config(
                    app,
                    log_level="critical",
                    # exit promptly on SIGTERM at teardown — don't hang waiting on the long-lived
                    # MCP SSE connection to close (the parent SIGKILLs us / PR_SET_PDEATHSIG fires).
                    timeout_graceful_shutdown=0,
                )
            ).serve(sockets=[sock])
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


def serve_forked(app, sock: socket.socket, server) -> None:
    """Serve `app` on the bound `sock` as a fork-per-rollout multiplexer (see module docstring).
    `app` is the warm MCP ASGI app, built once in the parent after `setup`; forked children inherit
    it copy-on-write. `server` is the `ServerBase` whose tools the app exposes — each child runs its
    `setup_task` for the rollout's task. `sock` is the socket the launcher reads the port back from."""
    import httpx
    import uvicorn

    base = tempfile.mkdtemp(prefix="vf_fork_")
    children: dict[str, _Child] = {}
    forking = asyncio.Lock()
    client = httpx.AsyncClient(timeout=None)

    def _spawn(key: str, state_url: str) -> _Child:
        # Bind the child's port HERE and keep it bound across the fork — the child serves on this exact
        # (inherited) socket, so the port is never released: no TOCTOU window. Fully synchronous (no
        # await), so it runs entirely under `forking`, which serializes the fork() from a single loop.
        child_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        child_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        child_sock.bind(("127.0.0.1", 0))  # 0 = an OS-assigned free port
        port = child_sock.getsockname()[1]
        # name the private dir by a hash, not `key` — `key` is the rollout's bearer secret, which
        # shouldn't appear as a filesystem path
        slug = hashlib.sha256(key.encode()).hexdigest()[:16] if key else "_default"
        cwd = os.path.join(base, slug)
        os.makedirs(cwd, exist_ok=True)
        pid = os.fork()
        if pid == 0:
            # child: serves on the inherited socket (setup_task first); never returns
            _serve_child(app, child_sock, cwd, server, state_url, key)
        child_sock.close()  # parent: the child owns the socket fd now
        return _Child(pid, port, cwd)

    async def ensure(key: str, state_url: str) -> _Child:
        # Fast path: a live child — no lock, no readiness probe.
        child = children.get(key)
        if child is not None and child.ready.is_set():
            return child
        # The lock is held ONLY for the synchronous fork()+register, never the readiness wait — so a
        # cold fork serializes other forks (fork-safety) but doesn't stall traffic to other children.
        async with forking:
            child = children.get(key)
            creating = child is None
            if creating:
                children[key] = child = _spawn(key, state_url)
        if not creating:
            # Another task is bringing this key up — wait for it, don't re-fork.
            await child.ready.wait()
            if (
                children.get(key) is not child
            ):  # reaped / failed to start meanwhile → retry
                return await ensure(key, state_url)
            return child
        # We created it: wait for it to serve OUTSIDE the lock, so other keys fork in parallel.
        try:
            await _wait_up(child.port)
        except BaseException:
            await reap(
                key
            )  # dead on arrival: drop it (wakes any waiter to re-fork), then propagate
            raise
        child.ready.set()
        # log only the pid (it correlates spawn<->reap) — the key is the rollout's bearer secret
        logger.info("fork: spawned child pid=%d", child.pid)
        return child

    async def reap(key: str) -> None:
        child = children.pop(key, None)
        if not child:
            return
        child.ready.set()  # wake any waiter blocked in `ensure` — it re-checks `children` and re-forks
        with contextlib.suppress(Exception):
            os.kill(child.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            os.waitpid(child.pid, 0)
        shutil.rmtree(child.cwd, ignore_errors=True)
        logger.info("fork: reaped child pid=%d", child.pid)

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
        params = parse_qs(scope.get("query_string", b"").decode())
        key = (params.get(STATE_SECRET_PARAM) or [""])[0]
        state_url = (params.get(STATE_URL_PARAM) or [""])[0]
        if scope["path"] == CLOSE_PATH:
            await reap(key)
            await _respond(send, 200, b"closed")
            return
        body: list[bytes] = []
        more = True
        while more:  # read the request body, then forward to the rollout's child
            msg = await receive()
            body.append(msg.get("body", b""))
            more = msg.get("more_body", False)
        # Bring the child up and stream its response back. Any failure here (child never came up,
        # `ensure` timed out, upstream errored mid-stream) must not leave the ASGI caller hanging:
        # send a clean 502 if nothing was sent yet, else end the started body so the client stops.
        started = False
        try:
            child = await ensure(key, state_url)
            headers = [
                (k.decode(), v.decode())
                for k, v in scope["headers"]
                if k.decode().lower() not in _HOP_BY_HOP
            ]
            qs = scope.get("query_string", b"").decode()
            url = f"http://127.0.0.1:{child.port}{scope['path']}" + (
                f"?{qs}" if qs else ""
            )
            req = client.build_request(
                scope["method"], url, headers=headers, content=b"".join(body)
            )
            resp = await client.send(req, stream=True)
            try:
                out = [
                    (k.encode(), v.encode())
                    for k, v in resp.headers.items()
                    if k.lower() not in _HOP_BY_HOP
                ]
                await send(
                    {
                        "type": "http.response.start",
                        "status": resp.status_code,
                        "headers": out,
                    }
                )
                started = True
                async for chunk in resp.aiter_raw():
                    await send(
                        {"type": "http.response.body", "body": chunk, "more_body": True}
                    )
                await send(
                    {"type": "http.response.body", "body": b"", "more_body": False}
                )
            finally:
                await resp.aclose()
        except Exception as exc:
            logger.warning("fork: proxy error forwarding to child: %s", exc)
            if started:
                with contextlib.suppress(Exception):
                    await send(
                        {"type": "http.response.body", "body": b"", "more_body": False}
                    )
            else:
                await _respond(send, 502, b"fork proxy error")

    async def _serve() -> None:
        # timeout_graceful_shutdown=0: exit promptly on SIGTERM (the runtime's teardown) instead of
        # hanging on the long-lived proxied SSE — so `finally` below SIGKILLs the children.
        try:
            await uvicorn.Server(
                uvicorn.Config(proxy, log_level="critical", timeout_graceful_shutdown=0)
            ).serve(sockets=[sock])
        finally:
            await client.aclose()

    try:
        asyncio.run(_serve())
    finally:
        for key in list(children):
            with contextlib.suppress(Exception):
                os.kill(children[key].pid, signal.SIGKILL)
        shutil.rmtree(base, ignore_errors=True)
