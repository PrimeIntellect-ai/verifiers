"""The request gate's adaptive limit: a provider's 429/529 lowers it, it climbs back to the
cap on its own, and the refused requests are retried behind the gate."""

import asyncio
import os
import time
from pathlib import Path
from typing import Self

import httpx
import pytest
from aiohttp import web
from openai import AsyncOpenAI

import verifiers.v1 as vf
from verifiers.v1 import request_gate as gate_module
from verifiers.v1.cli import gate as gate_cli
from verifiers.v1.clients.base import build_async_httpx
from verifiers.v1.configs.client import EvalClientConfig, RequestGateConfig
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.request_gate import GatedAsyncClient, ModelGate, gate_key, retry_after
from verifiers.v1.runtimes import ProgramResult

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
    outright."""

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
        await request.json()
        if self.reject_first > 0 or self.active >= self.limit:
            self.reject_first = max(0, self.reject_first - 1)
            self.rejected += 1
            headers = {"Retry-After": self.retry_after} if self.retry_after else {}
            return web.json_response(
                {"error": {"message": "Too many concurrent requests"}},
                status=429,
                headers=headers,
            )
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.latency)
            self.served += 1
            return web.json_response(COMPLETION)
        finally:
            self.active -= 1

    async def __aenter__(self) -> Self:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        self.url = f"http://127.0.0.1:{self.runner.addresses[0][1]}/v1"
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.runner.cleanup()


@pytest.fixture
def gates(tmp_path: Path) -> Path:
    """A fresh gate directory (mode 0700, as the gate requires)."""
    directory = tmp_path / "gates"
    directory.mkdir(mode=0o700)
    return directory


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
    (status,) = gate_module.gate_states(tmp_path / "gates")
    return status


def admit_all(gate: ModelGate, config: RequestGateConfig) -> list[int]:
    slots = []
    while (slot := gate.try_admit(config)) is not None:
        slots.append(slot)
    return slots


def refuse(gate: ModelGate, config: RequestGateConfig, times: int = 1) -> None:
    for _ in range(times):
        slot = gate.try_admit(config)
        assert slot is not None
        gate.release(slot, rejected=True)


# -- the limit, at the daemon ---------------------------------------------------------


def test_a_rejection_lowers_the_limit_once_per_cooldown(gates):
    config = RequestGateConfig(max_concurrent=8, decrease_cooldown=60)
    gate = ModelGate(gates, "k", "m")
    refuse(gate, config)
    assert gate.state()["limit"] == 6, "8 x 0.75"
    refuse(gate, config, times=3)
    status = gate.state()
    assert (status["limit"], status["rejections"]) == (6, 4), "one burst, one decrease"
    assert len(admit_all(gate, config)) == 6


def test_the_limit_never_drops_below_min_concurrent(gates):
    config = RequestGateConfig(max_concurrent=8, min_concurrent=3, decrease_cooldown=0)
    gate = ModelGate(gates, "k", "m")
    refuse(gate, config, times=20)
    assert gate.state()["limit"] == 3


def test_a_gate_that_is_not_adaptive_keeps_its_cap(gates):
    config = RequestGateConfig(max_concurrent=8, adaptive=False, decrease_cooldown=0)
    gate = ModelGate(gates, "k", "m")
    refuse(gate, config, times=5)
    status = gate.state()
    assert (status["limit"], status["rejections"]) == (8, 5)


def test_the_limit_climbs_back_to_the_cap_in_a_straight_line(gates):
    config = RequestGateConfig(max_concurrent=8, recover_seconds=2.0)
    gate = ModelGate(gates, "k", "m")
    refuse(gate, config)  # 8 -> 6 at t0
    started = time.monotonic()
    assert gate.state()["limit"] == 6
    time.sleep(max(0.0, 1.1 - (time.monotonic() - started)))
    assert gate.state()["limit"] == 7, "halfway back: 6 + (8 - 6) x 1/2"
    time.sleep(max(0.0, 2.1 - (time.monotonic() - started)))
    assert gate.state()["limit"] == 8
    assert len(admit_all(gate, config)) == 8


async def test_a_waiter_held_back_by_a_lowered_limit_is_admitted_as_it_recovers(
    gates,
):
    config = RequestGateConfig(max_concurrent=2, recover_seconds=0.5)
    gate = ModelGate(gates, "k", "m")
    refuse(gate, config)  # 2 -> 1.5: one may be in flight
    held = gate.try_admit(config)
    assert held is not None
    started = time.monotonic()
    second = await asyncio.wait_for(gate.acquire(config), 5)  # no release frees it
    assert time.monotonic() - started >= 0.2
    assert {held, second} == {0, 1}


def test_the_adaptive_limit_follows_the_cap(gates):
    config = RequestGateConfig(max_concurrent=8, decrease_cooldown=0)
    gate = ModelGate(gates, "k", "m")
    refuse(gate, config)
    assert gate.state()["limit"] == 6
    gate_module.resize(gates, 7)
    assert len(admit_all(gate, config)) == 6, "a lower keeps a learned limit below it"
    for slot in range(6):
        gate.release(slot, rejected=True)
    assert gate.state()["limit"] < 6, "429s still lower the limit"
    (raised,) = gate_module.resize(gates, 16)
    assert raised["limit"] == 16, "a raise restarts the limit at the new cap"


def test_a_learned_limit_lasts_only_while_a_process_uses_the_gate(gates):
    config = RequestGateConfig(max_concurrent=32, decrease_cooldown=0)
    first = ModelGate(gates, "k", "m")
    refuse(first, config, times=15)
    assert first.state()["limit"] == 1
    joining = ModelGate(gates, "k", "m")  # a run that starts while the first is live
    assert joining.state()["limit"] == 1, "a live user's limit is shared"
    first.close_inherited()  # both runs end: their descriptors close
    joining.close_inherited()
    later = ModelGate(gates, "k", "m")
    assert len(admit_all(later, config)) == 32, "a later run starts from its cap"


def test_a_forked_child_keeps_the_learned_limit(gates):
    config = RequestGateConfig(max_concurrent=2, decrease_cooldown=0)
    parent = gate_module.model_gate(gates, "fork", "m")
    refuse(parent, config)  # 2 -> 1.5
    held = parent.try_admit(config)
    assert held is not None
    pid = os.fork()
    if pid == 0:  # the child: report through the exit code, never return into pytest
        ok = False
        try:
            child = gate_module.model_gate(gates, "fork", "m")
            ok = child.state()["limit"] == 1 and child.try_admit(config) is None
        finally:
            os._exit(0 if ok else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    parent.release(held)


async def test_try_admit_never_jumps_the_queue(gates):
    """A lowered limit climbing back frees room before the daemon's recheck admits the
    queue's head; a `try_admit` in that gap must not take it."""
    config = RequestGateConfig(max_concurrent=2, recover_seconds=0.5)
    gate = ModelGate(gates, "k", "m")
    refuse(gate, config)  # 2 -> 1.5: one may be in flight
    held = gate.try_admit(config)
    assert held is not None
    waiter = asyncio.create_task(gate.acquire(config))
    jumped = []

    def trying() -> None:
        while not waiter.done():
            if (slot := gate.try_admit(config)) is not None:
                jumped.append(slot)
                return

    await asyncio.gather(asyncio.to_thread(trying), asyncio.wait_for(waiter, 10))
    assert jumped == [], "the waiter, first in line, got the freed room"


def test_min_concurrent_cannot_exceed_the_cap():
    with pytest.raises(ValueError):
        RequestGateConfig(max_concurrent=2, min_concurrent=3)


# -- the client -----------------------------------------------------------------------


async def test_the_gate_backs_off_under_a_lower_provider_limit_and_recovers(tmp_path):
    config = gate_config(
        tmp_path,
        max_concurrent=12,
        backoff=0.05,
        max_backoff=0.2,
        decrease_cooldown=0,
        recover_seconds=1.0,
    )
    async with FakeProvider(limit=3) as provider, client(provider, config) as http:
        responses = await asyncio.gather(*(post(http, provider.url) for _ in range(40)))
        assert all(r.status_code == 200 for r in responses), "every 429 was absorbed"
        assert provider.rejected > 0
        for _ in range(100):  # releases land on the daemon a moment later
            lowered = state(tmp_path)
            if lowered["rejections"] == provider.rejected:
                break
            await asyncio.sleep(0.01)
        assert lowered["rejections"] == provider.rejected
        assert lowered["limit"] < 12, "backed off"
        await asyncio.sleep(1.1)
        assert state(tmp_path)["limit"] == 12, "recovered to the cap"


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


def test_retry_after_reads_milliseconds_seconds_and_dates():
    def wait(**headers: str) -> float | None:
        return retry_after(httpx.Response(429, headers=headers))

    assert wait(**{"retry-after-ms": "250"}) == 0.25
    assert wait(**{"retry-after": "3"}) == 3.0
    assert wait(**{"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}) == 0.0
    assert wait() is None and wait(**{"retry-after": "soon"}) is None


async def test_the_last_rejection_reaches_the_caller(tmp_path):
    config = gate_config(tmp_path, max_concurrent=2, max_rejections=2, backoff=0.01)
    async with (
        FakeProvider(limit=2, reject_first=10) as provider,
        client(provider, config) as http,
    ):
        response = await post(http, provider.url)
    assert response.status_code == 429
    assert provider.rejected == 3, "two absorbed, the third returned"


async def test_a_gate_that_absorbs_none_passes_the_429_through(tmp_path):
    config = gate_config(tmp_path, max_concurrent=2, max_rejections=0)
    async with (
        FakeProvider(limit=2, reject_first=1) as provider,
        client(provider, config) as http,
    ):
        response = await post(http, provider.url)
    assert response.status_code == 429 and provider.rejected == 1
    for _ in range(100):
        if state(tmp_path)["rejections"] == 1:
            break
        await asyncio.sleep(0.01)
    assert state(tmp_path)["rejections"] == 1, "it still counts"


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
    gate = gate_module.model_gate(
        tmp_path / "gates", gate_key(request.url, "", MODEL), MODEL
    )
    for _ in range(100):  # the release lands on the daemon a moment later
        if (slot := gate.try_admit(config)) is not None:
            break
        await asyncio.sleep(0.02)
    assert slot == 0, "the only slot is free again"


def test_vf_gate_shows_the_live_limit_and_refusals():
    table = gate_cli.render(
        [
            {
                "model": MODEL,
                "max_concurrent": 32,
                "limit": 24,
                "rejections": 3,
                "in_flight": 24,
                "waiting": 7,
                "admitted": 100,
                "wait_seconds": 0.5,
                "updated": 10.0,
            }
        ],
        now=12.0,
    )
    header, row = (line.split() for line in table.splitlines())
    values = dict(zip(header, row))
    assert (values["cap"], values["limit"], values["429s"]) == ("32", "24", "3")
    assert values["updated"] == "2s"


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
    """A 429 the gate absorbs never reaches the harness: its one call succeeds, even with
    no SDK retries."""
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


def gated_agent(provider: FakeProvider, tmp_path: Path, **gate) -> vf.Agent:
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
                    **gate,
                },
            },
            runtime={"type": "subprocess"},
            harness={"id": "null"},
        )
    )
    agent.harness = OneCallHarness(agent.harness.config)
    return agent


async def test_the_harness_sdk_does_not_retry_a_429_the_gate_gave_up_on(
    tmp_path, monkeypatch
):
    """After `max_rejections`, the refusal reaches the harness marked
    `x-should-retry: false`, so a retrying SDK makes no further attempt."""
    monkeypatch.setenv("FAKE_KEY", "k")
    async with FakeProvider(limit=2, reject_first=100, retry_after="0.01") as provider:
        agent = gated_agent(provider, tmp_path, max_rejections=1)
        agent.harness.sdk_retries = 3
        trace = await agent.run(vf.Task(vf.TaskData(idx=0, prompt="go")))
    assert not trace.ok
    assert provider.rejected == 2, "one absorbed by the gate, one returned, no more"


async def test_a_gate_that_absorbs_none_leaves_429s_to_the_harness_sdk(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_KEY", "k")
    async with FakeProvider(limit=2, reject_first=100, retry_after="0.01") as provider:
        agent = gated_agent(provider, tmp_path, max_rejections=0)
        agent.harness.sdk_retries = 3
        trace = await agent.run(vf.Task(vf.TaskData(idx=0, prompt="go")))
    assert not trace.ok
    assert provider.rejected == 4, "the SDK's first try and its three retries"
