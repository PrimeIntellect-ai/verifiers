"""The in-box ACP runner's provider relay resends what the Prime tunnel gateway turns away
while the tunnel's frpc logs in again, and passes everything else through unchanged."""

import importlib.util
import sys
import types
from pathlib import Path

import httpx
import pytest
from aiohttp import web

import verifiers.v1.acp as acp_harness

GATEWAY_404 = (
    "<!DOCTYPE html><html><head><title>404 - Tunnel Not Found</title></head>"
    "<body>Tunnel Not Found</body></html>"
)


@pytest.fixture(scope="module")
def runner():
    """runner.py is a uv script run inside the box; its `acp` dependency is not installed
    here, and the relay doesn't use it, so a stub stands in."""
    stub = types.ModuleType("acp")
    for name in (
        "PROTOCOL_VERSION",
        "Client",
        "RequestError",
        "image_block",
        "spawn_agent_process",
        "text_block",
    ):
        setattr(stub, name, type(name, (), {}))
    schema = types.ModuleType("acp.schema")
    schema.__getattr__ = lambda name: type(name, (), {})  # type: ignore[attr-defined]
    saved = {name: sys.modules.get(name) for name in ("acp", "acp.schema")}
    sys.modules.update({"acp": stub, "acp.schema": schema})
    try:
        path = Path(acp_harness.__file__).parent / "runner.py"
        spec = importlib.util.spec_from_file_location("vf_acp_runner", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, module_before in saved.items():
            if module_before is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module_before
    return module


async def serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"http://127.0.0.1:{port}"


async def relayed(runner_module, upstream: str):
    relay = runner_module.ProviderRelay(f"{upstream}/v1")
    await relay.start()
    env = relay.route({"CONFIG": f'{{"baseURL": "{upstream}/v1"}}', "OTHER": "x"})
    assert env == {"CONFIG": f'{{"baseURL": "{relay.local_origin}/v1"}}', "OTHER": "x"}
    return relay


async def test_a_gateway_404_is_resent_until_the_tunnel_is_back(runner, monkeypatch):
    """The gateway's `Tunnel Not Found` 404 means the request never reached the
    interception server: the relay resends it, and the agent gets the streamed answer."""
    monkeypatch.setattr(runner, "RELAY_RETRY_SECONDS", 10.0)
    bodies: list[bytes] = []

    async def completions(request: web.Request) -> web.StreamResponse:
        bodies.append(await request.read())
        if len(bodies) <= 2:
            return web.Response(status=404, text=GATEWAY_404, content_type="text/html")
        assert request.headers["Authorization"] == "Bearer secret"
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for part in (b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"):
            await response.write(part)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    server, upstream = await serve(app)
    relay = await relayed(runner, upstream)
    body = b'{"model": "m", "stream": true, "messages": []}' * 1000
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{relay.local_origin}/v1/chat/completions",
                content=body,
                headers={"Authorization": "Bearer secret"},
            )
    finally:
        await relay.close()
        await server.cleanup()
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content == b"data: one\n\ndata: two\n\ndata: [DONE]\n\n"
    assert bodies == [body] * 3


async def test_other_answers_pass_through_without_a_resend(runner):
    """A 404 or error of the interception server's own is the agent's to see, once."""
    hits: list[str] = []

    async def missing(request: web.Request) -> web.Response:
        hits.append(request.path)
        return web.json_response({"error": "no such route"}, status=404)

    async def failing(request: web.Request) -> web.Response:
        hits.append(request.path)
        return web.json_response({"error": "upstream 503"}, status=503)

    app = web.Application()
    app.router.add_post("/v1/missing", missing)
    app.router.add_post("/v1/failing", failing)
    server, upstream = await serve(app)
    relay = await relayed(runner, upstream)
    try:
        async with httpx.AsyncClient() as client:
            missed = await client.post(f"{relay.local_origin}/v1/missing", json={})
            failed = await client.post(f"{relay.local_origin}/v1/failing", json={})
    finally:
        await relay.close()
        await server.cleanup()
    assert missed.status_code == 404
    assert missed.json() == {"error": "no such route"}
    assert failed.status_code == 503
    assert hits == ["/v1/missing", "/v1/failing"]


async def test_the_gateway_404_reaches_the_agent_once_the_retry_budget_is_spent(
    runner, monkeypatch
):
    monkeypatch.setattr(runner, "RELAY_RETRY_SECONDS", 0.6)
    hits = 0

    async def gone(request: web.Request) -> web.Response:
        nonlocal hits
        hits += 1
        return web.Response(status=404, text=GATEWAY_404, content_type="text/html")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", gone)
    server, upstream = await serve(app)
    relay = await relayed(runner, upstream)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{relay.local_origin}/v1/chat/completions", json={}
            )
    finally:
        await relay.close()
        await server.cleanup()
    assert response.status_code == 404
    assert "Tunnel Not Found" in response.text
    assert hits == 2  # sent, resent after 0.25 s; the next 0.5 s wait passes the budget


def test_the_host_hands_the_runner_its_endpoint():
    """The session's prompt config names the endpoint the runner's relay fronts."""
    source = Path(acp_harness.__file__).read_text()
    assert '"provider_url": self.endpoint' in source
