import asyncio
import time

import httpx
import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.errors import ProviderError
from verifiers.v1.interception.server import (
    RLM_LINEAGE_METADATA_KEY,
    InterceptionServer,
    _rlm_lineage,
)
from verifiers.v1.session import RolloutSession
from verifiers.v1.types import AssistantMessage, Response, Usage


def _trace() -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=7, prompt="solve it"),
        ),
    )


def _context() -> ModelContext:
    return ModelContext(
        model="test-model",
        client=EvalClientConfig(base_url="http://unused.invalid", api_key_var="NONE"),
    )


def _lineage_headers(call_id: str = "call-1") -> dict[str, str]:
    return {
        "X-RLM-Lineage-Version": "1",
        "X-RLM-Session-ID": "trace-1",
        "X-RLM-Invocation-ID": "invocation-1",
        "X-RLM-Segment-ID": "segment-1",
        "X-RLM-Call-ID": call_id,
        "X-RLM-Depth": "0",
        "X-RLM-Call-Kind": "turn",
    }


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
            "id": "completion-1",
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
            id="completion-1",
            created=int(time.time()),
            model="test-model",
            message=AssistantMessage(content="done"),
            finish_reason="stop",
            usage=Usage(prompt_tokens=2, completion_tokens=1),
            raw=raw,
        )


class _FailOnceClient(_BlockingClient):
    async def get_response(self, dialect, body, sampling, **kwargs) -> Response:
        if self.calls == 0:
            self.calls += 1
            raise ProviderError("transient", status_code=503)
        return await super().get_response(dialect, body, sampling, **kwargs)


def test_rlm_lineage_parser_handles_parents_compaction_and_future_versions():
    headers = httpx.Headers(
        {
            **_lineage_headers(),
            "X-RLM-Parent-Invocation-ID": "parent-invocation",
            "X-RLM-Parent-Call-ID": "parent-call",
            "X-RLM-Depth": "2",
            "X-RLM-Call-Kind": "compaction",
        }
    )

    call_id, metadata = _rlm_lineage(headers)

    assert call_id == "call-1"
    assert metadata[RLM_LINEAGE_METADATA_KEY]["parent_invocation_id"] == (
        "parent-invocation"
    )
    assert metadata[RLM_LINEAGE_METADATA_KEY]["parent_call_id"] == "parent-call"
    assert metadata[RLM_LINEAGE_METADATA_KEY]["depth"] == 2
    assert metadata[RLM_LINEAGE_METADATA_KEY]["call_kind"] == "compaction"
    future = httpx.Headers({**_lineage_headers(), "X-RLM-Lineage-Version": "2"})
    assert _rlm_lineage(future) == (None, {})
    invalid = httpx.Headers({**_lineage_headers(), "X-RLM-Depth": "not-an-int"})
    with pytest.raises(ValueError, match="depth"):
        _rlm_lineage(invalid)


@pytest.mark.asyncio
async def test_rlm_call_identity_coalesces_and_records_lineage_without_forwarding_it():
    trace = _trace()
    session = RolloutSession(ctx=_context(), trace=trace)
    client = _BlockingClient()
    session.client = client
    body = {"model": "ignored", "messages": [{"role": "user", "content": "hi"}]}

    async with InterceptionServer() as server:
        secret = "rollout-secret"
        server.sessions[secret] = session
        headers = {"Authorization": f"Bearer {secret}", **_lineage_headers()}
        async with httpx.AsyncClient() as http:
            first = asyncio.create_task(
                http.post(
                    f"{server.base_url}/v1/chat/completions", json=body, headers=headers
                )
            )
            await client.started.wait()
            second = asyncio.create_task(
                http.post(
                    f"{server.base_url}/v1/chat/completions", json=body, headers=headers
                )
            )
            await asyncio.sleep(0)
            client.release.set()
            first_response, second_response = await asyncio.gather(first, second)
            replay = await http.post(
                f"{server.base_url}/v1/chat/completions", json=body, headers=headers
            )
            reused = await http.post(
                f"{server.base_url}/v1/chat/completions",
                json={**body, "temperature": 0.5},
                headers=headers,
            )

    assert first_response.status_code == second_response.status_code == 200
    assert replay.status_code == 200
    assert reused.status_code == 400
    assert client.calls == 1
    assert all(not name.lower().startswith("x-rlm-") for name in client.headers)
    assert len(trace.calls) == 1
    assert trace.calls[0].metadata[RLM_LINEAGE_METADATA_KEY] == {
        "version": 1,
        "session_id": "trace-1",
        "invocation_id": "invocation-1",
        "parent_invocation_id": None,
        "segment_id": "segment-1",
        "call_id": "call-1",
        "parent_call_id": None,
        "depth": 0,
        "call_kind": "turn",
    }
    assert trace.num_branches == 1


@pytest.mark.asyncio
async def test_distinct_rlm_call_ids_with_identical_bodies_both_sample():
    trace = _trace()
    session = RolloutSession(ctx=_context(), trace=trace)
    client = _BlockingClient()
    client.release.set()
    session.client = client
    body = {"model": "ignored", "messages": [{"role": "user", "content": "hi"}]}

    async with InterceptionServer() as server:
        secret = "rollout-secret"
        server.sessions[secret] = session
        async with httpx.AsyncClient() as http:
            for call_id in ("call-1", "call-2"):
                response = await http.post(
                    f"{server.base_url}/v1/chat/completions",
                    json=body,
                    headers={
                        "Authorization": f"Bearer {secret}",
                        **_lineage_headers(call_id),
                    },
                )
                assert response.status_code == 200

    assert client.calls == 2
    assert [
        call.metadata[RLM_LINEAGE_METADATA_KEY]["call_id"] for call in trace.calls
    ] == ["call-1", "call-2"]


@pytest.mark.asyncio
async def test_failed_rlm_call_can_retry_with_the_same_identity():
    trace = _trace()
    session = RolloutSession(ctx=_context(), trace=trace)
    client = _FailOnceClient()
    client.release.set()
    session.client = client
    body = {"model": "ignored", "messages": [{"role": "user", "content": "hi"}]}

    async with InterceptionServer() as server:
        secret = "rollout-secret"
        server.sessions[secret] = session
        headers = {"Authorization": f"Bearer {secret}", **_lineage_headers()}
        async with httpx.AsyncClient() as http:
            failed = await http.post(
                f"{server.base_url}/v1/chat/completions", json=body, headers=headers
            )
            succeeded = await http.post(
                f"{server.base_url}/v1/chat/completions", json=body, headers=headers
            )

    assert failed.status_code == 503
    assert succeeded.status_code == 200
    assert client.calls == 2
    assert len(trace.calls) == 2
    assert trace.calls[0].error is not None
    assert trace.calls[1].node is not None
