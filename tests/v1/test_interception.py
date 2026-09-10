"""The interception server's request admission: `max_concurrent_requests` bounds the
upstream model requests in flight across a server's sessions, streamed responses holding
their slot until the upstream connection closes."""

import asyncio
import json

import httpx
import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.clients.client import Client, RelayReply
from verifiers.v1.interception.server import (
    InterceptionServer,
    InterceptionServerConfig,
)
from verifiers.v1.session import RolloutSession

COMPLETION = {
    "id": "cmpl",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
CHUNK = {
    "id": "cmpl",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
}


def sse(data: dict | bytes) -> bytes:
    payload = data if isinstance(data, bytes) else json.dumps(data).encode()
    return b"data: " + payload + b"\n\n"


class GatedClient(Client):
    """Counts upstream requests in flight; each holds until `release` is set."""

    def __init__(self) -> None:
        self.inflight = 0
        self.peak = 0
        self.release = asyncio.Event()
        self.arrived = asyncio.Condition()

    async def _enter(self) -> None:
        async with self.arrived:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            self.arrived.notify_all()

    async def get_response(
        self, dialect, body, sampling, session_id=None, turn=None, headers=None
    ):
        await self._enter()
        try:
            await self.release.wait()
        finally:
            self.inflight -= 1
        response = dialect.parse_response(dialect.validate_response(COMPLETION))
        response.raw = COMPLETION
        return response

    async def relay(self, dialect, body, session_id=None, headers=None):
        await self._enter()

        async def chunks():
            yield sse(CHUNK)
            await self.release.wait()
            yield sse(b"[DONE]")

        async def close() -> None:
            self.inflight -= 1  # the connection, not the last chunk, ends the request

        return RelayReply(
            content_type="text/event-stream", chunks=chunks(), close=close
        )


def make_session(idx: int) -> RolloutSession:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=idx, prompt="hi")),
    )
    return RolloutSession(
        ctx=ModelContext(model="m", client=EvalClientConfig()), trace=trace
    )


@pytest.mark.parametrize("stream", [False, True], ids=["rendered", "streamed"])
async def test_max_concurrent_requests_bounds_upstream(stream):
    upstream = GatedClient()
    config = InterceptionServerConfig(max_concurrent_requests=2)
    async with InterceptionServer(config) as server:
        sessions = [make_session(idx) for idx in range(5)]
        secrets = []
        for session in sessions:
            model_secret, _ = server.register(session)
            session.client = upstream
            secrets.append(model_secret)

        async with httpx.AsyncClient(base_url=server.base_url, timeout=10) as client:

            async def call(secret: str) -> httpx.Response:
                body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
                return await client.post(
                    "/v1/chat/completions",
                    json={**body, "stream": stream},
                    headers={"Authorization": f"Bearer {secret}"},
                )

            calls = [asyncio.create_task(call(secret)) for secret in secrets]
            async with upstream.arrived:
                await asyncio.wait_for(
                    upstream.arrived.wait_for(lambda: upstream.inflight >= 2), 5
                )
            await asyncio.sleep(0.1)  # the other three must still be queued
            assert upstream.inflight == 2 and not any(call.done() for call in calls)
            upstream.release.set()
            responses = await asyncio.gather(*calls)

    assert [response.status_code for response in responses] == [200] * 5
    assert upstream.peak == 2
    assert [len(session.trace.calls) for session in sessions] == [1] * 5
    assert [session.trace.num_turns for session in sessions] == [1] * 5
