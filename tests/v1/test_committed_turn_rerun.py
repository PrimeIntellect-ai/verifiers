"""A streamed turn that fails after its SSE stream is committed is rerun by the
interception (`_buffered_stream(rerun=...)`), since harness SDKs do not retry an SSE
error event; only a provider failure before the turn committed is. Served over a real
aiohttp server with the timings shrunk; no model, but for the scripted endpoint of
`test_agent_resume`."""

# ruff: noqa: F811 - the fixtures imported from test_agent_resume are parameters

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from openai import AsyncOpenAI
from test_agent_resume import (  # noqa: F401 - fixtures
    TOOLS,
    NotesTask,
    ScriptHarness,
    fake_boxes,
    make,
    model,
)

import verifiers.v1 as vf
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.interception import server
from verifiers.v1.runtimes import ProgramResult


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


def provider_turns(*statuses: int):
    """`turns`, its failures marked as the provider's before anything committed."""
    produce, calls = turns(*statuses)

    async def marked() -> web.Response:
        response = await produce()
        response[server.RERUNNABLE] = response.status >= 400
        return response

    return marked, calls


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
    produce, calls = provider_turns(502, 503, 200)
    body = await serve(produce, rerun=True)
    assert calls == [502, 503, 200]
    assert body.endswith(b"data: ok\n\ndata: [DONE]\n\n")


async def test_a_committed_4xx_is_final():
    produce, calls = provider_turns(400, 200)
    body = await serve(produce, rerun=True)
    assert calls == [400]
    assert json.loads(body.split(b"data: ", 1)[1])["error"]["message"] == "upstream 400"


async def test_reruns_stop_after_the_attempt_budget():
    produce, calls = provider_turns(500)
    body = await serve(produce, rerun=True)
    assert len(calls) == server.COMMITTED_TURN_ATTEMPTS
    assert b"upstream 500" in body


async def test_without_a_rerun_the_failure_is_served_once():
    produce, calls = turns(429, 200)
    body = await serve(produce, rerun=False)
    assert calls == [429]
    assert b"upstream 429" in body


async def test_a_committed_failure_that_is_not_the_providers_is_final():
    """A failure `sample` does not mark rerunnable (a hook failing after the turn
    committed) is served as it is: sampling again would commit another turn."""
    produce, calls = turns(502, 200)
    body = await serve(produce, rerun=True)
    assert calls == [502]
    assert b"upstream 502" in body


class StreamHarness(ScriptHarness):
    """One streamed model call, read to its end."""

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
    ) -> ProgramResult:
        _, prompt = self.resolve_prompt(data)
        client = AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0)
        try:
            stream = await client.chat.completions.create(
                model=ctx.model,
                messages=[{"role": "user", "content": prompt}],
                tools=TOOLS,
                stream=True,
            )
            async for _ in stream:
                pass
        finally:
            await client.close()
        return ProgramResult(exit_code=0, stdout="", stderr="")


class GatedTask(NotesTask):
    """Its request hook fails on the tool-call gate, after the reply committed."""

    @vf.intercept
    async def gate(self, request: vf.Request) -> None:
        if request.messages[-1].role == "tool":
            raise RuntimeError("the gate broke")


@pytest.fixture
async def streaming(model):
    """The scripted endpoint, answering a streamed request with its reply as SSE."""
    from test_agent_resume import reply

    async def handle(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        model.requests.append(body)
        message = reply(body["messages"])["choices"][0]["message"]
        calls = [
            {"index": i, **call} for i, call in enumerate(message.get("tool_calls", []))
        ]
        chunk = {
            "id": "scripted",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "scripted",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "tool_calls": calls},
                    "finish_reason": "tool_calls",
                }
            ],
        }
        events = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
        return web.Response(body=events, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    async with TestServer(app) as test_server:
        model.url = str(test_server.make_url("/v1"))
        yield model


async def test_a_hook_failing_after_the_commit_is_not_sampled_again(
    streaming, monkeypatch
):
    model = streaming
    monkeypatch.setattr(server, "KEEPALIVE_GRACE_SECONDS", 0.0)  # commit at once
    agent = make(model, harness=StreamHarness)
    trace = await agent.run(GatedTask(vf.TaskData(idx=0, prompt="start")))
    assert not trace.ok
    assert "the gate broke" in trace.errors[-1].message
    assert len(model.requests) == 1
    assert len([node for node in trace.nodes if node.sampled]) == 1
