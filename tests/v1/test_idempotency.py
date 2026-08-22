import asyncio
import time

import httpx
import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.errors import ProviderError
from verifiers.v1.interception import server as interception_server
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


class _FailingBlockingClient:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_response(self, dialect, body, sampling, **kwargs) -> Response:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        raise ProviderError("rate limited", status_code=429)


def _session() -> tuple[vf.Trace, RolloutSession]:
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
    return trace, session


@pytest.mark.asyncio
async def test_idempotency_key_coalesces_and_replays_one_model_call():
    trace, session = _session()
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


@pytest.mark.asyncio
async def test_idempotency_key_coalesces_original_error_without_caching_it():
    _, session = _session()
    client = _FailingBlockingClient()
    session.client = client
    body = {"model": "ignored", "messages": [{"role": "user", "content": "hi"}]}

    async with InterceptionServer() as server:
        secret = "rollout-secret"
        server.sessions[secret] = session
        headers = {
            "Authorization": f"Bearer {secret}",
            "idempotency-key": "call-error",
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
            while len(session.tasks) < 2:
                await asyncio.sleep(0)
            client.release.set()
            first_response, concurrent_response = await asyncio.gather(
                first, concurrent_retry
            )
            assert client.calls == 1
            later_retry = await http.post(
                f"{server.base_url}/v1/chat/completions", json=body, headers=headers
            )

    assert first_response.status_code == 429
    assert concurrent_response.status_code == 429
    assert concurrent_response.content == first_response.content
    assert later_retry.status_code == 429
    assert client.calls == 2
    assert "call-error" not in session.idempotent_requests


@pytest.mark.asyncio
async def test_idempotency_cache_bounds_completed_responses(monkeypatch):
    monkeypatch.setattr(interception_server, "IDEMPOTENCY_CACHE_MAX_COMPLETED", 1)
    _, session = _session()
    client = _BlockingClient()
    client.release.set()
    session.client = client
    body = {"model": "ignored", "messages": [{"role": "user", "content": "hi"}]}

    async with InterceptionServer() as server:
        secret = "rollout-secret"
        server.sessions[secret] = session
        headers = {"Authorization": f"Bearer {secret}"}
        async with httpx.AsyncClient() as http:
            first = await http.post(
                f"{server.base_url}/v1/chat/completions",
                json=body,
                headers={**headers, "idempotency-key": "call-1"},
            )
            await http.post(
                f"{server.base_url}/v1/chat/completions",
                json=body,
                headers={**headers, "idempotency-key": "call-2"},
            )
            assert set(session.idempotent_requests) == {"call-2"}
            evicted_retry = await http.post(
                f"{server.base_url}/v1/chat/completions",
                json=body,
                headers={**headers, "idempotency-key": "call-1"},
            )

    assert evicted_retry.json() != first.json()
    assert client.calls == 3
    assert set(session.idempotent_requests) == {"call-1"}
