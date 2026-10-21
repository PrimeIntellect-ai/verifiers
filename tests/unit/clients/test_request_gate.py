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
from verifiers.v1.interception import server
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
    for _ in range(100):  # releases on an event loop land on the gate's thread
        final = state(tmp_path)
        if final.get("rejections") == provider.rejected:
            break
        await asyncio.sleep(0.02)
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
    slot = gate.try_admit(RequestGateConfig(max_concurrent=1))
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
        assert gate.try_admit(config) is None
    finally:
        os.kill(holder.pid, signal.SIGKILL)
        holder.wait()
    assert gate.try_admit(config) == 0


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
    from verifiers.v1.interception.server import _retry_wait

    def wait(status: int, gated: bool) -> float | None:
        error = vf.ProviderError("x", status_code=status)
        return _retry_wait(error, 0, 0.0, vf.ProviderRetryConfig(), gated)

    assert wait(429, gated=False) and wait(529, gated=False)
    assert wait(429, gated=True) is None and wait(529, gated=True) is None
    assert wait(502, gated=True) and wait(503, gated=False)
    assert wait(400, gated=False) is None


async def test_the_harness_sdk_does_not_retry_a_429_the_gate_gave_up_on(
    tmp_path, monkeypatch
):
    """The gate is the only 429 retry layer: the interception does not retry its final
    rejection, which reaches the harness marked `x-should-retry: false`, so a retrying
    SDK makes no further attempt."""
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
    for _ in range(100):  # a release on an event loop lands on the gate's thread
        if (slot := gate.try_admit(config)) is not None:
            break
        await asyncio.sleep(0.02)
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


def test_a_learned_limit_lasts_only_while_a_process_uses_the_gate(tmp_path):
    config = RequestGateConfig(max_concurrent=32, decrease_cooldown=0)

    def limit() -> float:
        return json.loads(first.state_path.read_text())["limit"]

    first = ModelGate(tmp_path, "k", "m")
    for _ in range(15):
        slot = first.try_admit(config)
        assert slot is not None
        first.release(slot, config, "rejected")
    assert limit() == 1
    joining = ModelGate(tmp_path, "k", "m")  # a run that starts while the first is live
    assert limit() == 1, "a live user's limit is shared"
    first.close_inherited()  # both runs end: their descriptors close
    joining.close_inherited()
    later = ModelGate(tmp_path, "k", "m")
    slots = []
    while (slot := later.try_admit(config)) is not None:
        slots.append(slot)
    assert len(slots) == 32, "a later run starts from max_concurrent"


async def test_a_gate_that_absorbs_no_rejections_leaves_429s_to_the_interception(
    tmp_path, monkeypatch
):
    """With `max_rejections = 0` the gate retries no 429, so the interception still
    does: a gated run never retries 429s less than an ungated one."""
    monkeypatch.setenv("FAKE_KEY", "k")
    monkeypatch.setattr(server, "RETRY_BACKOFF_SECONDS", 0.01)
    async with FakeProvider(limit=2, reject_first=2, retry_after="0.01") as provider:
        agent = vf.Agent(
            vf.AgentConfig(
                model=MODEL,
                client={
                    "type": "eval",
                    "base_url": provider.url,
                    "api_key_var": "FAKE_KEY",
                    "request_gate": {
                        "max_concurrent": 2,
                        "max_rejections": 0,
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
    assert provider.rejected == 2, "both 429s were retried in place"
    assert [call.error is None for call in trace.calls] == [False, False, True]


def test_a_forked_child_drops_the_parents_permits_but_keeps_the_learned_limit(tmp_path):
    config = RequestGateConfig(max_concurrent=2, decrease_cooldown=0)
    parent = gate_module.model_gate(tmp_path, "fork", "m")
    rejected = parent.try_admit(config)
    assert rejected is not None
    parent.release(rejected, config, "rejected")
    held = parent.try_admit(config)
    assert held is not None
    pid = os.fork()
    if pid == 0:  # the child: report through the exit code, never return into pytest
        ok = False
        try:
            child = gate_module.model_gate(tmp_path, "fork", "m")
            lowered = json.loads(child.state_path.read_text())["lowered"]
            ok = (
                child is not parent
                and not child._held
                and lowered == 1.5  # not started afresh: the parent still uses the gate
                and child.try_admit(config) is None  # the parent's permit still counts
            )
        finally:
            os._exit(0 if ok else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    parent.release(held, config, "other")
    assert parent.try_admit(config) is not None, "the child took nothing with it"


def admit_all(gate: ModelGate, config: RequestGateConfig) -> list[int]:
    slots = []
    while (slot := gate.try_admit(config)) is not None:
        slots.append(slot)
    return slots


def test_an_operator_raises_the_cap_of_a_live_gate_for_every_user(tmp_path):
    config = RequestGateConfig(max_concurrent=2, adaptive=False)
    first, second = ModelGate(tmp_path, "k", "m"), ModelGate(tmp_path, "k", "m")
    assert len(admit_all(first, config) + admit_all(second, config)) == 2
    (raised,) = gate_module.resize(tmp_path, 5)
    assert (raised["max_concurrent"], raised["limit"]) == (5, 5)
    taken = admit_all(second, config) + admit_all(first, config)
    assert sorted(taken) == [2, 3, 4], "the raise reached both, under one shared cap"
    (status,) = gate_module.gate_states(tmp_path)
    assert (status["model"], status["max_concurrent"]) == ("m", 5)
    assert (status["limit"], status["in_flight"]) == (5, 5)


def test_a_lowered_cap_sheds_nothing_and_admits_once_below_it(tmp_path):
    config = RequestGateConfig(max_concurrent=4, adaptive=False)
    first, second = ModelGate(tmp_path, "k", "m"), ModelGate(tmp_path, "k", "m")
    low = [first.try_admit(config), first.try_admit(config)]
    high = admit_all(second, config)
    assert (low, high) == ([0, 1], [2, 3])
    gate_module.resize(tmp_path, 2)
    assert first.try_admit(config) is None and second.try_admit(config) is None
    for slot in low:
        first.release(slot, config, "ok")
    assert first.try_admit(config) is None, "the two left above the cap still count"
    second.release(high[0], config, "ok")
    assert first.try_admit(config) == 0, "one in flight: below the cap of 2"
    assert second.try_admit(config) is None
    second.release(high[1], config, "ok")
    assert second.try_admit(config) == 1
    assert json.loads(first.state_path.read_text())["slots"] == 2, "back to the cap"


def test_a_smaller_config_admitting_first_never_shrinks_the_shared_cap(tmp_path):
    small, large = ModelGate(tmp_path, "k", "m"), ModelGate(tmp_path, "k", "m")
    assert len(admit_all(small, RequestGateConfig(max_concurrent=2))) == 2
    assert len(admit_all(large, RequestGateConfig(max_concurrent=6))) == 4
    (status,) = gate_module.gate_states(tmp_path)
    assert status["max_concurrent"] == 6
    assert small.try_admit(RequestGateConfig(max_concurrent=2)) is None, "at the cap"


def test_a_joining_user_admits_under_the_shared_cap_and_a_fresh_one_under_its_config(
    tmp_path,
):
    first = ModelGate(tmp_path, "k", "m")
    assert len(admit_all(first, RequestGateConfig(max_concurrent=2))) == 2
    gate_module.resize(tmp_path, 3)
    joining = ModelGate(tmp_path, "k", "m")
    assert admit_all(joining, RequestGateConfig(max_concurrent=8)) == [2]
    first.close_inherited()
    joining.close_inherited()
    fresh = ModelGate(tmp_path, "k", "m")  # no user left: the operator's cap is gone
    assert len(admit_all(fresh, RequestGateConfig(max_concurrent=6))) == 6


def test_the_adaptive_limit_follows_the_cap(tmp_path):
    config = RequestGateConfig(max_concurrent=8, decrease_cooldown=0)
    gate = ModelGate(tmp_path, "k", "m")

    def limit() -> float:
        return json.loads(gate.state_path.read_text())["limit"]

    slot = gate.try_admit(config)
    assert slot is not None
    gate.release(slot, config, "rejected")
    assert limit() == 6
    gate_module.resize(tmp_path, 7)
    assert len(admit_all(gate, config)) == 6, "a lower keeps a learned limit below it"
    for slot in range(6):
        gate.release(slot, config, "rejected")
    assert limit() < 6, "429s still shrink the limit"
    (raised,) = gate_module.resize(tmp_path, 16)
    assert raised["limit"] == 16, "a raise restarts the limit at the new cap"
    gate_module.resize(tmp_path, 2)
    for _ in range(20):
        slot = gate.try_admit(config)
        assert slot is not None
        gate.release(slot, config, "ok")
    assert limit() == 2, "successes grow the limit only up to the cap"


def test_resize_picks_gates_by_model_and_refuses_a_zero_cap(tmp_path):
    config = RequestGateConfig(max_concurrent=4)
    for model in ("a", "b"):
        ModelGate(tmp_path, model, model).try_admit(config)
    (resized,) = gate_module.resize(tmp_path, 9, model="b")
    assert resized["model"] == "b"
    caps = {s["model"]: s["max_concurrent"] for s in gate_module.gate_states(tmp_path)}
    assert caps == {"a": 4, "b": 9}
    assert gate_module.resize(tmp_path / "none", 9) == []
    with pytest.raises(ValueError):
        gate_module.resize(tmp_path, 0)


RESIZED_CHILD = textwrap.dedent(
    f"""
    import asyncio, sys
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
                for _ in range(30)
            ))
        assert all(r.status_code == 200 for r in responses), [r.status_code for r in responses]

    asyncio.run(main(sys.argv[1], sys.argv[2]))
    """
)


async def test_a_resize_reaches_running_processes(tmp_path):
    """Two processes started with a cap of 2 share a raise to 4 made while they run."""
    gates = tmp_path / "gates"
    async with FakeProvider(limit=10, latency=0.1) as provider:
        children = [
            await asyncio.create_subprocess_exec(
                sys.executable, "-c", RESIZED_CHILD, provider.url, str(gates)
            )
            for _ in range(2)
        ]
        while provider.served < 4:
            await asyncio.sleep(0.02)
        assert provider.peak == 2
        (resized,) = gate_module.resize(gates, 4)
        assert resized["max_concurrent"] == 4
        codes = await asyncio.wait_for(
            asyncio.gather(*(child.wait() for child in children)), 60
        )
    assert codes == [0, 0]
    assert provider.served == 60 and provider.peak == 4 and provider.rejected == 0


async def test_a_request_waiting_on_a_busy_gate_lock_never_stalls_its_event_loop(
    tmp_path,
):
    # Every process on the host flocks the gate's state for each admission. Taken on the
    # event loop, a contended lock stalled every coroutine of a worker (its box polls among
    # them) for as long as another process held it: box setups blew their timeouts.
    import fcntl

    gate = ModelGate(tmp_path, "k", "m")
    config = RequestGateConfig(max_concurrent=1, adaptive=False)
    other = os.open(
        gate.state_path, os.O_RDWR | os.O_CREAT, 0o600
    )  # another process's lock
    fcntl.flock(other, fcntl.LOCK_EX)
    ticks = 0

    async def ticking():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(ticking())
    admission = asyncio.create_task(gate.acquire(config))
    await asyncio.sleep(0.5)
    assert not admission.done() and ticks >= 20  # the loop ran while the lock was held
    fcntl.flock(other, fcntl.LOCK_UN)
    os.close(other)
    slot = await asyncio.wait_for(admission, 5)
    ticker.cancel()
    gate.release(slot, config, "ok")  # applied on the gate's thread
    for _ in range(100):
        if gate.try_admit(config) is not None:
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("the released slot never came back")


async def test_a_cancelled_admission_returns_the_slot_its_gate_thread_took(tmp_path):
    import fcntl

    gate = ModelGate(tmp_path, "k", "m")
    config = RequestGateConfig(max_concurrent=1, adaptive=False)
    other = os.open(gate.state_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(other, fcntl.LOCK_EX)
    admission = asyncio.create_task(gate.acquire(config))
    await asyncio.sleep(0.2)
    admission.cancel()
    fcntl.flock(other, fcntl.LOCK_UN)
    os.close(other)
    with pytest.raises(asyncio.CancelledError):
        await admission
    for _ in range(
        100
    ):  # the gate thread finishes the admission, then gives the slot back
        if not gate._held:
            break
        await asyncio.sleep(0.05)
    assert gate._held == set()
