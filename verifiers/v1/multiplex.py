"""Automatic per-rollout isolation for a shared tool server, by fork.

`run_mcp_server(mcp, multiplex=True)` warms the server's (expensive) setup ONCE in a parent
process, then forks a child per rollout id on first contact: the child inherits the warm state
copy-on-write (so in-memory state is isolated on write) and runs in a private working directory
(so relative-path on-disk writes are isolated too). The parent is a thin async reverse proxy
that pins each rollout id to its child and forwards MCP traffic (streaming, so SSE works).
Children are reaped on a `POST /vf/close?rollout_id=...` from the framework (rollout teardown),
or by an idle TTL.

Caveats: Linux/fork only; do NOT use with CUDA/GPU state or background threads in the server
(fork copies neither safely) — fork happens from a single-threaded loop here, before any such
state. Writes to absolute paths outside the private CWD are not isolated.
"""

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import tempfile
import time
from urllib.parse import parse_qs

from verifiers.v1.tools import ROLLOUT_ID_PARAM, _free_port

logger = logging.getLogger(__name__)

CLOSE_PATH = "/vf/close"  # framework POSTs here on rollout teardown to reap the child
_IDLE_TTL = float(os.environ.get("VF_MULTIPLEX_TTL", "900"))  # reap idle children after N s
_HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host"}


class _Child:
    def __init__(self, pid: int, port: int, cwd: str) -> None:
        self.pid, self.port, self.cwd = pid, port, cwd
        self.last = time.monotonic()


def _serve_child(app, port: int, cwd: str) -> None:
    """In the forked child: move to a private CWD and serve `app` on `port` with a fresh loop
    (the inherited parent loop is abandoned). Never returns."""
    import uvicorn

    os.chdir(cwd)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            uvicorn.Server(
                uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical")
            ).serve()
        )
    finally:
        os._exit(0)


async def _wait_up(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _, w = await asyncio.open_connection("127.0.0.1", port)
            w.close()
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise RuntimeError(f"multiplex child on port {port} did not come up")


def run_multiplexed(mcp) -> None:
    """Serve `mcp` on `MCP_PORT` as a fork-per-rollout multiplexer (see module docstring)."""
    import httpx
    import uvicorn

    parent_port = int(os.environ["MCP_PORT"])
    app = mcp.streamable_http_app()  # built once in the warm parent; forked children inherit it
    base = tempfile.mkdtemp(prefix="vf_mplex_")
    children: dict[str, _Child] = {}
    forking = asyncio.Lock()
    client = httpx.AsyncClient(timeout=None)

    async def ensure(rid: str) -> _Child:
        async with forking:  # one fork at a time keeps the single-threaded fork clean
            child = children.get(rid)
            if child is None:
                port = _free_port()
                cwd = os.path.join(base, rid or "_default")
                os.makedirs(cwd, exist_ok=True)
                pid = os.fork()
                if pid == 0:
                    _serve_child(app, port, cwd)  # child: never returns
                children[rid] = child = _Child(pid, port, cwd)
                await _wait_up(port)
                logger.info("multiplex: forked child pid=%d for rollout %s", pid, rid)
            child.last = time.monotonic()
            return child

    async def reap(rid: str) -> None:
        child = children.pop(rid, None)
        if not child:
            return
        with contextlib.suppress(Exception):
            os.kill(child.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            os.waitpid(child.pid, 0)
        shutil.rmtree(child.cwd, ignore_errors=True)
        logger.info("multiplex: reaped child pid=%d for rollout %s", child.pid, rid)

    async def _send_response(send, status: int, body: bytes) -> None:
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": body})

    async def proxy(scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            # Drive the proxy app's own (empty) lifespan; children run their own.
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        rid = (parse_qs(scope.get("query_string", b"").decode()).get(ROLLOUT_ID_PARAM) or [""])[0]
        if scope["path"] == CLOSE_PATH:
            await reap(rid)
            await _send_response(send, 200, b"closed")
            return
        # Read the request body, forward to the rollout's child, stream the response back.
        body, more = b"", True
        while more:
            msg = await receive()
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
        child = await ensure(rid)
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
        await send({"type": "http.response.start", "status": resp.status_code, "headers": out})
        async for chunk in resp.aiter_raw():
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        await resp.aclose()

    async def _reaper() -> None:
        while True:
            await asyncio.sleep(30)
            now = time.monotonic()
            for rid in [r for r, c in children.items() if now - c.last > _IDLE_TTL]:
                await reap(rid)

    async def _serve() -> None:
        asyncio.ensure_future(_reaper())
        await uvicorn.Server(
            uvicorn.Config(proxy, host="127.0.0.1", port=parent_port, log_level="critical")
        ).serve()

    try:
        asyncio.run(_serve())
    finally:
        for rid in list(children):
            child = children[rid]
            with contextlib.suppress(Exception):
                os.kill(child.pid, signal.SIGKILL)
        shutil.rmtree(base, ignore_errors=True)
