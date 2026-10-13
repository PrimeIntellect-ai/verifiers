"""A streamed turn that fails after its SSE stream is committed is rerun by the
interception (`_buffered_stream(rerun=...)`), since harness SDKs do not retry an SSE
error event. Served over a real aiohttp server with the timings shrunk; no model."""

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.interception import server


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(server, "KEEPALIVE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(server, "KEEPALIVE_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(server, "COMMITTED_TURN_BACKOFF_SECONDS", 0.0)


def turns(*statuses: int):
    """A turn producer answering with `statuses` in order (each after a short wait, so
    the stream is committed first), counting its calls."""
    calls: list[int] = []

    async def produce() -> web.Response:
        await asyncio.sleep(0.05)
        status = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(status)
        if status >= 400:
            return web.json_response(
                {"error": {"message": f"upstream {status}"}}, status=status
            )
        return web.Response(
            body=b"data: ok\n\ndata: [DONE]\n\n", content_type="text/event-stream"
        )

    return produce, calls


async def serve(produce, *, rerun: bool) -> bytes:
    async def handler(request: web.Request) -> web.StreamResponse:
        return await server._buffered_stream(
            request,
            ChatDialect(),
            produce(),
            "trace-1",
            rerun=produce if rerun else None,
        )

    app = web.Application()
    app.router.add_post("/", handler)
    async with TestServer(app) as test_server, aiohttp.ClientSession() as client:
        response = await client.post(test_server.make_url("/"))
        assert response.status == 200  # committed before the turn finished
        return await response.read()


async def test_a_committed_5xx_is_rerun_until_it_succeeds():
    produce, calls = turns(502, 503, 200)
    body = await serve(produce, rerun=True)
    assert calls == [502, 503, 200]
    assert body.endswith(b"data: ok\n\ndata: [DONE]\n\n")


async def test_a_committed_4xx_is_final():
    produce, calls = turns(400, 200)
    body = await serve(produce, rerun=True)
    assert calls == [400]
    assert json.loads(body.split(b"data: ", 1)[1])["error"]["message"] == "upstream 400"


async def test_reruns_stop_after_the_attempt_budget():
    produce, calls = turns(500)
    body = await serve(produce, rerun=True)
    assert len(calls) == server.COMMITTED_TURN_ATTEMPTS
    assert b"upstream 500" in body


async def test_without_a_rerun_the_failure_is_served_once():
    produce, calls = turns(429, 200)
    body = await serve(produce, rerun=False)
    assert calls == [429]
    assert b"upstream 429" in body
