import asyncio
import time

import httpx
import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.session import RolloutSession
from verifiers.v1.types import AssistantMessage, Response, Usage


class _BlockingClient:
    def __init__(self) -> None:
        self.calls = 0
        self.headers: dict[str, str] = {}
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_response(self, dialect, body, sampling, **kwargs) -> Response:
        self.calls += 1
        self.headers = dict(kwargs["headers"])
        self.started.set()
        await self.release.wait()
        raw = {
            "id": f"completion-{self.calls}",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }
        return Response(
            id=raw["id"],
            created=int(time.time()),
            model="test-model",
            message=AssistantMessage(content="done"),
            finish_reason="stop",
            usage=Usage(prompt_tokens=2, completion_tokens=1),
            raw=raw,
        )


@pytest.mark.asyncio
async def test_idempotency_key_coalesces_and_replays_one_model_call():
    trace = vf.Trace(
        id="trace-1",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=7, prompt="solve it"),
        ),
    )
    session = RolloutSession(
        ctx=ModelContext(
            model="test-model",
            client=EvalClientConfig(
                base_url="http://unused.invalid", api_key_var="NONE"
            ),
        ),
        trace=trace,
    )
    client = _BlockingClient()
    session.client = client
    body = {"model": "ignored", "messages": [{"role": "user", "content": "hi"}]}

    async with InterceptionServer() as server:
        secret = "rollout-secret"
        server.sessions[secret] = session
        headers = {
            "Authorization": f"Bearer {secret}",
            "idempotency-key": "call-1",
        }
        async with httpx.AsyncClient() as http:
            first = asyncio.create_task(
                http.post(
                    f"{server.base_url}/v1/chat/completions", json=body, headers=headers
                )
            )
            await client.started.wait()
            concurrent_retry = asyncio.create_task(
                http.post(
                    f"{server.base_url}/v1/chat/completions", json=body, headers=headers
                )
            )
            await asyncio.sleep(0)
            client.release.set()
            first_response, concurrent_response = await asyncio.gather(
                first, concurrent_retry
            )
            completed_retry = await http.post(
                f"{server.base_url}/v1/chat/completions", json=body, headers=headers
            )
            conflicting_retry = await http.post(
                f"{server.base_url}/v1/chat/completions",
                json={**body, "temperature": 0.5},
                headers=headers,
            )
            distinct_call = await http.post(
                f"{server.base_url}/v1/chat/completions",
                json=body,
                headers={**headers, "idempotency-key": "call-2"},
            )

    assert first_response.status_code == 200
    assert concurrent_response.json() == first_response.json()
    assert completed_retry.json() == first_response.json()
    assert conflicting_retry.status_code == 400
    assert distinct_call.status_code == 200
    assert client.calls == 2
    provider_keys = [
        value
        for name, value in client.headers.items()
        if name.lower() == "idempotency-key"
    ]
    assert len(provider_keys) == 1
    assert provider_keys[0] != "call-2"
    assert len(trace.calls) == 2
