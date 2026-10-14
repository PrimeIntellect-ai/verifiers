"""The request gate against a fake provider that enforces a concurrency limit with 429s."""

import asyncio
import gc
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import httpx
import pytest
from aiohttp import web
from openai import AsyncOpenAI

import verifiers.v1 as vf
from verifiers.v1.clients import gate as gate_module
from verifiers.v1.clients.base import build_async_httpx
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.clients.gate import GatedAsyncClient, ModelGate, gate_key
from verifiers.v1.configs.client import EvalClientConfig, RequestGateConfig
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.types import Sampling

MODEL = "internal/fake-model"
COMPLETION = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class FakeProvider:
    """Answers chat completions after `latency`, refusing with 429 above `limit` in flight
    (with `Retry-After: retry_after` when set). `reject_first` refuses that many requests
    outright. Streams SSE when the request asks to."""

    def __init__(
        self,
        limit: int,
        latency: float = 0.05,
        retry_after: str | None = None,
        reject_first: int = 0,
    ) -> None:
        self.limit, self.latency, self.retry_after = limit, latency, retry_after
        self.reject_first = reject_first
        self.active = self.peak = self.rejected = self.served = 0
        self.url = ""

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if self.reject_first > 0 or self.active >= self.limit:
            self.reject_first = max(0, self.reject_first - 1)
            self.rejected += 1
            headers = {"Retry-After": self.retry_after} if self.retry_after else {}
            return web.json_response(
                {"error": {"message": "Too many concurrent requests (limit: 3)"}},
                status=429,
                headers=headers,
            )
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if not body.get("stream"):
                await asyncio.sleep(self.latency)
                self.served += 1
                return web.json_response(COMPLETION)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for _ in range(3):
                await asyncio.sleep(self.latency)
                await response.write(b'data: {"choices": []}\n\n')
            await response.write(b"data: [DONE]\n\n")
            self.served += 1
            return response
        finally:
            self.active -= 1

    async def __aenter__(self) -> Self:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]
        self.url = f"http://127.0.0.1:{port}/v1"
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.runner.cleanup()


def gate_config(tmp_path: Path, **kwargs) -> RequestGateConfig:
    return RequestGateConfig(dir=str(tmp_path / "gates"), **kwargs)


def client(provider: FakeProvider, gate: RequestGateConfig) -> httpx.AsyncClient:
    return build_async_httpx(
        EvalClientConfig(
            base_url=provider.url, api_key_var="FAKE_KEY", request_gate=gate
        )
    )


async def post(http: httpx.AsyncClient, url: str) -> httpx.Response:
    return await http.post(
        f"{url}/chat/completions",
        json={"model": MODEL, "messages": []},
        headers={"Authorization": "Bearer k"},
    )


def state(tmp_path: Path) -> dict:
    (path,) = (tmp_path / "gates").glob("*.state.json")
    return json.loads(path.read_text())


async def test_a_static_cap_at_the_provider_limit_draws_no_429(tmp_path):
    async with (
        FakeProvider(limit=3) as provider,
        client(provider, gate_config(tmp_path, max_concurrent=3)) as http,
    ):
        responses = await asyncio.gather(*(post(http, provider.url) for _ in range(20)))
    assert [r.status_code for r in responses] == [200] * 20
    assert provider.rejected == 0 and provider.peak == 3
    assert state(tmp_path)["in_flight"] <= 3


async def test_the_adaptive_limit_shrinks_under_a_lower_provider_limit(tmp_path):
    config = gate_config(
        tmp_path, max_concurrent=12, backoff=0.05, max_backoff=0.2, decrease_cooldown=0
    )
    async with FakeProvider(limit=3) as provider, client(provider, config) as http:
        responses = await asyncio.gather(*(post(http, provider.url) for _ in range(40)))
    assert all(r.status_code == 200 for r in responses), "every 429 was absorbed"
    assert provider.rejected > 0
    final = state(tmp_path)
    assert final["rejections"] == provider.rejected
    assert final["limit"] < 12


async def test_retry_after_is_honoured(tmp_path):
    config = gate_config(tmp_path, max_concurrent=4)
    async with (
        FakeProvider(limit=4, retry_after="0.3", reject_first=1) as provider,
        client(provider, config) as http,
    ):
        started = time.monotonic()
        response = await post(http, provider.url)
        elapsed = time.monotonic() - started
    assert response.status_code == 200
    assert elapsed >= 0.3
    assert provider.rejected == 1


async def test_the_last_rejection_reaches_the_caller(tmp_path):
    config = gate_config(tmp_path, max_concurrent=2, max_rejections=2, backoff=0.01)
    async with (
        FakeProvider(limit=2, reject_first=10) as provider,
        client(provider, config) as http,
    ):
        response = await post(http, provider.url)
    assert response.status_code == 429
    assert provider.rejected == 3, "two absorbed, the third returned"


async def test_a_stream_holds_its_permit_until_closed(tmp_path):
    config = gate_config(tmp_path, max_concurrent=1, adaptive=False)
    async with FakeProvider(limit=5) as provider, client(provider, config) as http:
        request = http.build_request(
            "POST",
            f"{provider.url}/chat/completions",
            json={"model": MODEL, "messages": [], "stream": True},
            headers={"Authorization": "Bearer k"},
        )
        streamed = await http.send(request, stream=True)
        second = asyncio.create_task(post(http, provider.url))
        await asyncio.sleep(0.3)
        assert not second.done(), "the open stream holds the only permit"
        await streamed.aread()
        await streamed.aclose()
        assert (await asyncio.wait_for(second, 5)).status_code == 200
    assert provider.peak == 1


async def test_requests_without_a_model_pass_ungated(tmp_path):
    config = gate_config(tmp_path, max_concurrent=1)
    async with FakeProvider(limit=5) as provider, client(provider, config) as http:
        response = await http.post(f"{provider.url}/chat/completions", json={})
    assert response.status_code == 200
    assert not (tmp_path / "gates").exists() or not list((tmp_path / "gates").iterdir())


async def test_the_eval_client_passes_the_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "k")
    async with FakeProvider(limit=2, reject_first=1) as provider:
        config = EvalClientConfig(
            base_url=provider.url,
            api_key_var="FAKE_KEY",
            request_gate=gate_config(tmp_path, max_concurrent=2, backoff=0.01),
        )
        eval_client = EvalClient(config)
        try:
            response = await eval_client.get_response(
                ChatDialect(),
                {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
                Sampling(),
            )
        finally:
            await eval_client.close()
    assert response.message.content == "ok"
    assert provider.rejected == 1, "the gate absorbed the 429"
    assert state(tmp_path)["rejections"] == 1, "and counted it"


CHILD = textwrap.dedent(
    f"""
    import asyncio, sys
    import httpx
    from verifiers.v1.clients.base import build_async_httpx
    from verifiers.v1.configs.client import EvalClientConfig, RequestGateConfig

    async def main(url, gates):
        config = EvalClientConfig(
            base_url=url,
            api_key_var="FAKE_KEY",
            request_gate=RequestGateConfig(max_concurrent=2, adaptive=False, dir=gates),
        )
        async with build_async_httpx(config) as http:
            responses = await asyncio.gather(*(
                http.post(url + "/chat/completions", json={{"model": "{MODEL}", "messages": []}},
                          headers={{"Authorization": "Bearer k"}})
                for _ in range(8)
            ))
        assert all(r.status_code == 200 for r in responses), [r.status_code for r in responses]

    asyncio.run(main(sys.argv[1], sys.argv[2]))
    """
)


async def test_processes_on_a_host_share_one_budget(tmp_path):
    async with FakeProvider(limit=10, latency=0.1) as provider:
        children = [
            await asyncio.create_subprocess_exec(
                sys.executable, "-c", CHILD, provider.url, str(tmp_path / "gates")
            )
            for _ in range(3)
        ]
        codes = await asyncio.wait_for(
            asyncio.gather(*(child.wait() for child in children)), 60
        )
    assert codes == [0, 0, 0]
    assert provider.served == 24 and provider.peak == 2 and provider.rejected == 0


HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from verifiers.v1.clients.gate import ModelGate
    from verifiers.v1.configs.client import RequestGateConfig

    gate = ModelGate(Path(sys.argv[1]), "k", "m")
    slot, _ = gate.try_admit(RequestGateConfig(max_concurrent=1))
    assert slot == 0
    print("held", flush=True)
    time.sleep(60)
    """
)


def test_a_killed_process_returns_its_permits(tmp_path):
    gates = tmp_path / "gates"
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(gates)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline() == "held\n"
        gate = ModelGate(gates, "k", "m")
        config = RequestGateConfig(max_concurrent=1)
        assert gate.try_admit(config) == (None, 1)
    finally:
        os.kill(holder.pid, signal.SIGKILL)
        holder.wait()
    slot, _ = gate.try_admit(config)
    assert slot == 0


def test_gates_are_keyed_by_endpoint_credential_and_model():
    url = httpx.URL("https://api.example/v1/chat/completions")
    other = httpx.URL("https://other.example/v1/chat/completions")
    key = gate_key(url, "Bearer a", MODEL)
    assert key == gate_key(
        httpx.URL("https://api.example/v1/messages"), "Bearer a", MODEL
    )
    assert (
        len(
            {
                key,
                gate_key(other, "Bearer a", MODEL),
                gate_key(url, "Bearer b", MODEL),
                gate_key(url, "Bearer a", "x"),
            }
        )
        == 4
    )


def test_min_concurrent_cannot_exceed_the_cap():
    with pytest.raises(ValueError):
        RequestGateConfig(max_concurrent=2, min_concurrent=3)


class OneCallHarness(NullHarness):
    """One chat completion through the interception endpoint, then done."""

    SUPPORTS_MCP = False
    SUPPORTS_TOOL_INTERCEPTION = False
    sdk_retries = 0

    async def setup(self, runtime: vf.Runtime) -> None:
        return None

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
    ) -> ProgramResult:
        client = AsyncOpenAI(
            base_url=endpoint, api_key=secret, max_retries=self.sdk_retries
        )
        try:
            await client.chat.completions.create(
                model=ctx.model, messages=[{"role": "user", "content": "hi"}]
            )
        finally:
            await client.close()
        return ProgramResult(exit_code=0, stdout="", stderr="")


async def test_absorbed_429s_never_reach_the_harness(tmp_path, monkeypatch):
    """A 429 the gate absorbs never reaches the harness: its call succeeds."""
    monkeypatch.setenv("FAKE_KEY", "k")
    async with FakeProvider(limit=2, reject_first=2) as provider:
        agent = vf.Agent(
            vf.AgentConfig(
                model=MODEL,
                client={
                    "type": "eval",
                    "base_url": provider.url,
                    "api_key_var": "FAKE_KEY",
                    "request_gate": {
                        "max_concurrent": 2,
                        "backoff": 0.01,
                        "dir": str(tmp_path / "gates"),
                    },
                },
                runtime={"type": "subprocess"},
                harness={"id": "null"},
            )
        )
        agent.harness = OneCallHarness(agent.harness.config)
        trace = await agent.run(vf.Task(vf.TaskData(idx=0, prompt="go")))
    assert trace.ok, trace.errors
    (call,) = trace.calls
    assert call.error is None
    assert provider.rejected == 2


def test_a_gated_rejection_is_not_retried_again():
    from verifiers.v1.interception.server import _retryable

    assert _retryable(429, gated=False) and _retryable(529, gated=False)
    assert not _retryable(429, gated=True) and not _retryable(529, gated=True)
    assert _retryable(502, gated=True) and _retryable(503, gated=False)
    assert not _retryable(400, gated=False)


async def test_the_harness_sdk_does_not_retry_a_429_the_gate_gave_up_on(
    tmp_path, monkeypatch
):
    """The gate is the only 429 retry layer: its final rejection reaches the harness
    marked `x-should-retry: false`, so a retrying SDK makes no further attempt."""
    monkeypatch.setenv("FAKE_KEY", "k")
    async with FakeProvider(limit=2, reject_first=100) as provider:
        agent = vf.Agent(
            vf.AgentConfig(
                model=MODEL,
                client={
                    "type": "eval",
                    "base_url": provider.url,
                    "api_key_var": "FAKE_KEY",
                    "request_gate": {
                        "max_concurrent": 2,
                        "max_rejections": 1,
                        "backoff": 0.01,
                        "dir": str(tmp_path / "gates"),
                    },
                },
                runtime={"type": "subprocess"},
                harness={"id": "null"},
            )
        )
        agent.harness = OneCallHarness(agent.harness.config)
        agent.harness.sdk_retries = 3
        trace = await agent.run(vf.Task(vf.TaskData(idx=0, prompt="go")))
    assert not trace.ok
    assert provider.rejected == 2, "one absorbed by the gate, one returned, no more"


async def test_a_rejection_cancelled_while_closing_returns_its_permit(tmp_path):
    """A retried 429 whose close is cancelled (a rollout timeout) still frees its slot."""
    closing = asyncio.Event()

    class SlowClose(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"busy"

        async def aclose(self):
            closing.set()
            await asyncio.Event().wait()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "0"}, stream=SlowClose())

    config = gate_config(tmp_path, max_concurrent=1)
    http = GatedAsyncClient(config, transport=httpx.MockTransport(handler))
    request = http.build_request("POST", "http://x/v1", json={"model": MODEL})
    sending = asyncio.create_task(http.send(request, stream=True))
    await closing.wait()
    sending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sending
    gate = ModelGate(tmp_path / "gates", gate_key(request.url, "", MODEL), MODEL)
    slot, _ = gate.try_admit(config)
    assert slot == 0, "the only slot is free again"


def test_a_stream_collected_unclosed_inside_the_gate_does_not_deadlock(
    tmp_path, monkeypatch
):
    """An unclosed streamed response is freed by the cyclic garbage collector, which can
    run inside the gate's own mutex section; its finalizer's release must not wait on
    that mutex. The requests run in a thread so a deadlock fails the test, not the run."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(b"{}"))

    collect = False
    real_loads = json.loads

    def loads(raw):  # the collector firing at an allocation inside `try_admit`
        if collect:
            gc.collect()
        return real_loads(raw)

    monkeypatch.setattr(
        gate_module, "json", SimpleNamespace(loads=loads, dumps=json.dumps)
    )
    config = gate_config(tmp_path, max_concurrent=1)

    async def main() -> int:
        nonlocal collect
        http = GatedAsyncClient(config, transport=httpx.MockTransport(handler))
        request = http.build_request("POST", "http://x/v1", json={"model": MODEL})
        streamed = await http.send(request, stream=True)
        del streamed  # never closed: only the cyclic collector frees it
        collect = True
        request = http.build_request("POST", "http://x/v1", json={"model": MODEL})
        return (await http.send(request)).status_code

    statuses: list[int] = []
    gc.disable()
    try:
        thread = threading.Thread(
            target=lambda: statuses.append(asyncio.run(main())), daemon=True
        )
        thread.start()
        thread.join(10)
    finally:
        gc.enable()
    assert not thread.is_alive(), "the gate deadlocked on its own mutex"
    assert statuses == [200], "the collected stream's permit was returned"
