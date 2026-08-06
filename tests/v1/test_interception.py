"""Interception-server tests: a provider stream with an out-of-enum finish label.

Regression for the trace-commit loss: the SDK's closed `finish_reason` Literal used to
make `ChatStreamParser.finish()` raise AFTER the stream had been relayed — the client
got HTTP 200 and the full stream, but the turn never committed (node=None, no usage,
an errored ModelCall). The dialect now absorbs the label, so the turn must commit.
"""

import httpx
from aiohttp import web

import verifiers.v1 as vf
from verifiers.v1.clients import resolve_client
from verifiers.v1.clients.client import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession

UP_CHUNKS = [
    (
        b'data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,'
        b'"model":"test-model","choices":[{"index":0,'
        b'"delta":{"role":"assistant","content":"Hello"}}]}\n\n'
    ),
    (
        b'data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":1,'
        b'"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"error"}],'
        b'"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
    ),
    b"data: [DONE]\n\n",
]


async def test_stream_with_out_of_enum_finish_reason_commits_turn():
    """A fake upstream streams a terminal chunk with `finish_reason: "error"`: the
    client still receives the full stream AND the turn commits to the trace."""

    async def upstream(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"}
        )
        await resp.prepare(request)
        for chunk in UP_CHUNKS:
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    up_app = web.Application()
    up_app.router.add_post("/chat/completions", upstream)
    up_runner = web.AppRunner(up_app)
    await up_runner.setup()
    up_site = web.TCPSite(up_runner, "127.0.0.1", 0)
    await up_site.start()
    up_port = up_site._server.sockets[0].getsockname()[1]

    client_cfg = EvalClientConfig(
        base_url=f"http://127.0.0.1:{up_port}", api_key_var="NO_SUCH_KEY_VAR"
    )
    session = RolloutSession(
        ctx=ModelContext(model="test-model", client=client_cfg),
        client=resolve_client(client_cfg),
        trace=vf.Trace(
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")),
        ),
    )
    try:
        async with (
            InterceptionServer(requires_tunnel=False) as server,
            server.acquire(session) as (base_url, model_secret, _state_secret),
        ):
            received = b""
            async with (
                httpx.AsyncClient(timeout=30) as http,
                http.stream(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {model_secret}"},
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                ) as reply,
            ):
                assert reply.status_code == 200
                async for chunk in reply.aiter_bytes():
                    received += chunk

            # The client received the full stream, including the terminal event.
            assert b'"finish_reason":"error"' in received
            assert received.rstrip().endswith(b"data: [DONE]")

            # ... and the turn committed to the trace despite the out-of-enum label.
            assert session.trace.num_turns == 1
            assert len(session.trace.calls) == 1
            call = session.trace.calls[0]
            assert call.error is None
            assert call.node is not None
            assert call.finish_reason is None  # unknown label -> None
            assert call.usage is not None
    finally:
        await up_runner.cleanup()
