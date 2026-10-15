"""The Docker callback relay forwards capability URLs to host-loopback origins and
refuses everything else."""

import aiohttp
import pytest
from aiohttp import web

from verifiers.v1.runtimes.docker.callbacks import CallbackRelay


@pytest.fixture
async def origin():
    async def echo(request: web.Request) -> web.Response:
        body = await request.read()
        response = web.Response(body=body, headers={"X-Host": request.host})
        response.set_cookie("session", "1", path="/")
        return response

    async def moved(request: web.Request) -> web.Response:
        raise web.HTTPFound("/echo")

    app = web.Application()
    app.router.add_post("/echo", echo)
    app.router.add_get("/moved", moved)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@pytest.fixture
async def relay():
    relay = CallbackRelay()
    await relay.start("127.0.0.1")
    try:
        yield relay
    finally:
        await relay.stop()


async def test_relay_forwards_a_callback(origin, relay) -> None:
    url = relay.callback_url(f"{origin}/echo", "127.0.0.1")
    assert url.startswith(f"http://127.0.0.1:{relay.port}/.vf-host/")
    async with (
        aiohttp.ClientSession() as session,
        session.post(url, data=b"ping") as response,
    ):
        assert response.status == 200
        assert await response.read() == b"ping"
        assert response.headers["X-Host"] == origin.removeprefix("http://")
        cookie = response.headers["Set-Cookie"]
    token = url.split("/.vf-host/")[1].split("/")[0]
    assert cookie.startswith(f"{token}-session=1")
    assert f"Path=/.vf-host/{token}/" in cookie


async def test_relay_keeps_redirects_on_the_callback(origin, relay) -> None:
    url = relay.callback_url(f"{origin}/moved", "127.0.0.1")
    async with (
        aiohttp.ClientSession() as session,
        session.get(url, allow_redirects=False) as response,
    ):
        assert response.status == 302
        location = response.headers["Location"]
    assert location == relay.callback_url(f"{origin}/echo", "127.0.0.1")


async def test_relay_refuses_unknown_targets(relay) -> None:
    base = f"http://127.0.0.1:{relay.port}"
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base}/.vf-host/unknown/") as response:
            assert response.status == 403
        async with session.get(f"{base}/anything") as response:
            assert response.status == 403


def test_relay_only_routes_loopback_origins() -> None:
    with pytest.raises(ValueError, match="unsupported Docker host callback URL"):
        CallbackRelay().callback_url("https://example.com/", "127.0.0.1")
