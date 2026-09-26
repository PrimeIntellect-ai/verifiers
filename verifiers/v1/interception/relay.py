"""In-runtime relay in front of a tunneled interception server.

A remote runtime reaches the host's interception server through a tunnel. When the host's
tunnel client loses its connection to the tunnel service (an outage on the host's network),
the public URL goes dark until the client reconnects: the service answers ``404 Tunnel not
found or no longer active`` and connections fail. Programs in the runtime see that 404
directly — the host never does — and an OpenAI client, like most HTTP clients, does not
retry a 404, so every rollout behind that tunnel fails at once, whatever its harness.

`serve_relay` starts a loopback relay inside the runtime; programs reach the host through
it instead of the tunnel URL. The relay forwards each request unchanged and streams the
response back, retrying — for up to `RELAY_RETRY_SECONDS` — only failures that leave the
request unanswered by the host: the tunnel service's 404 and connection errors. A retried
model turn is safe: the interception server keeps a turn running when its reader
disconnects and coalesces the retried request onto it.
"""

import asyncio
import contextlib
import uuid

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import Runtime

RELAY_RETRY_SECONDS = 180.0
"""How long the relay retries one request while the tunnel is dark."""

RELAY_PROGRAM = r'''# /// script
# requires-python = ">=3.11"
# dependencies = ["aiohttp>=3.9"]
# ///
"""Loopback relay to a tunneled interception server; see verifiers.v1.interception.relay."""
import asyncio
import sys
import time

from aiohttp import ClientConnectionError, ClientSession, ClientTimeout, web

UPSTREAM, PORT_FILE, WINDOW = sys.argv[1].rstrip("/"), sys.argv[2], float(sys.argv[3])
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailer", "transfer-encoding", "upgrade", "host", "content-length",
}


def forwarded(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def tunnel_dark(status: int, body: bytes) -> bool:
    """The tunnel service's answer for a URL whose client is not connected."""
    return status == 404 and b"Tunnel not found" in body


async def relay(request: web.Request) -> web.StreamResponse:
    session: ClientSession = request.app["session"]
    body = await request.read()
    url = UPSTREAM + request.rel_url.raw_path_qs
    headers = forwarded(request.headers)
    deadline, delay = time.monotonic() + WINDOW, 0.5
    while True:
        failure = None
        try:
            upstream = await session.request(
                request.method, url, headers=headers, data=body, allow_redirects=False
            )
        except (ClientConnectionError, asyncio.TimeoutError) as e:
            failure = web.Response(status=502, text=f"relay: upstream unreachable: {e!r}")
        else:
            if upstream.status != 404:
                break
            payload = await upstream.read()
            reply = web.Response(status=404, body=payload, headers=forwarded(upstream.headers))
            if not tunnel_dark(404, payload):
                return reply
            failure = reply
        if time.monotonic() + delay > deadline:
            return failure
        print(f"relay: tunnel dark, retrying {request.method} {request.rel_url.path} in {delay:.1f}s", flush=True)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 10.0)
    response = web.StreamResponse(status=upstream.status, headers=forwarded(upstream.headers))
    await response.prepare(request)
    async for chunk in upstream.content.iter_any():
        await response.write(chunk)
    await response.write_eof()
    return response


async def main() -> None:
    app = web.Application(client_max_size=1 << 30)
    # No total timeout: model turns run as long as they run. Only connecting is bounded.
    app["session"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=30), auto_decompress=False)
    app.router.add_route("*", "/{path:.*}", relay)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    with open(PORT_FILE, "w") as f:
        f.write(str(site._server.sockets[0].getsockname()[1]))
    print(f"relay: 127.0.0.1 -> {UPSTREAM}", flush=True)
    await asyncio.Event().wait()


asyncio.run(main())
'''


async def serve_relay(runtime: Runtime, upstream: str) -> str:
    """Start a relay to `upstream` inside `runtime`; return its base URL there."""
    argv = await runtime.prepare_uv_script(RELAY_PROGRAM, activate=False)
    port_file = f"/tmp/vf-relay-port-{uuid.uuid4().hex}"
    log = f"vf_relay_{uuid.uuid4().hex[:8]}.log"
    await runtime.run_background(
        [*argv, upstream, port_file, str(RELAY_RETRY_SECONDS)], {}, log
    )
    for _ in range(120):
        with contextlib.suppress(Exception):
            data = (await runtime.read(port_file)).decode().strip()
            if data.isdigit():
                return f"http://127.0.0.1:{data}"
        await asyncio.sleep(0.5)
    raise SandboxError(f"interception relay did not start in the runtime (log: {log})")
