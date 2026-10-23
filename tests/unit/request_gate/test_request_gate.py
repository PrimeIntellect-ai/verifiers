"""The request gate on model clients, against a fake provider that counts its peak
concurrency and refuses with 429 above its limit."""

import asyncio
import contextlib
import gc
import json
import subprocess
import sys
import textwrap
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import httpx
import pytest
from aiohttp import web

import verifiers.v1 as vf
from verifiers.v1 import request_gate as gate_module
from verifiers.v1.cli import gate as gate_cli
from verifiers.v1.clients.base import build_async_httpx
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.configs.client import EvalClientConfig, RequestGateConfig
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.request_gate import GatedAsyncClient, gate_key
from verifiers.v1.types import Sampling

TEST_SECONDS = 120.0
"""Bound of a test that starts child processes (their start-up can stall on a loaded
host); a failing one prints the children's stderr."""

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
    """Answers chat completions after `latency`, refusing with 429 above `limit` in
    flight. Streams SSE when the request asks to."""

    def __init__(self, limit: int, latency: float = 0.05) -> None:
        self.limit, self.latency = limit, latency
        self.active = self.peak = self.rejected = self.served = 0
        self.models: list[str | None] = []
        self.url = ""

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.models.append(body.get("model"))
        if self.active >= self.limit:
            self.rejected += 1
            return web.json_response({"error": {"message": "busy"}}, status=429)
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
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        self.url = f"http://127.0.0.1:{self.runner.addresses[0][1]}/v1"
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.runner.cleanup()


def gate_config(tmp_path: Path, **kwargs) -> RequestGateConfig:
    return RequestGateConfig(dir=str(tmp_path / "gates"), **kwargs)


def client(provider: FakeProvider, gate: RequestGateConfig | None) -> httpx.AsyncClient:
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


async def test_no_gate_configured_sends_as_before(tmp_path):
    async with FakeProvider(limit=100) as provider, client(provider, None) as http:
        assert type(http) is httpx.AsyncClient
        responses = await asyncio.gather(*(post(http, provider.url) for _ in range(20)))
    assert [r.status_code for r in responses] == [200] * 20
    assert provider.peak == 20, "nothing holds requests back"


def test_a_saved_client_config_leaves_out_an_unset_gate():
    assert "request_gate" not in EvalClientConfig().model_dump(mode="json")
    gated = EvalClientConfig(request_gate={"max_concurrent": 2})
    assert gated.model_dump(mode="json")["request_gate"]["max_concurrent"] == 2


def test_an_agent_config_carries_the_gate_to_its_client():
    agent = vf.AgentConfig(
        client={"type": "eval", "request_gate": {"max_concurrent": 4}}
    )
    assert agent.client.request_gate == RequestGateConfig(max_concurrent=4)
    assert EvalClientConfig().request_gate is None, "off by default"


async def test_a_cap_at_the_provider_limit_draws_no_429(tmp_path):
    config = gate_config(tmp_path, max_concurrent=3)
    async with FakeProvider(limit=3) as provider, client(provider, config) as http:
        responses = await asyncio.gather(*(post(http, provider.url) for _ in range(30)))
    assert [r.status_code for r in responses] == [200] * 30
    assert provider.rejected == 0 and provider.peak == 3
    (state,) = gate_module.gate_states(tmp_path / "gates")
    assert (state["model"], state["admitted"], state["in_flight"]) == (MODEL, 30, 0)


async def test_a_stream_holds_its_permit_until_closed(tmp_path):
    config = gate_config(tmp_path, max_concurrent=1)
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
    assert not (tmp_path / "gates").exists(), "no gate, no daemon"


async def test_the_eval_client_passes_the_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "k")
    async with FakeProvider(limit=2) as provider:
        config = EvalClientConfig(
            base_url=provider.url,
            api_key_var="FAKE_KEY",
            request_gate=gate_config(tmp_path, max_concurrent=2),
        )
        eval_client = EvalClient(config)
        assert isinstance(eval_client.client, GatedAsyncClient)
        body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
        try:
            responses = await asyncio.gather(
                *(
                    eval_client.get_response(ChatDialect(), body, Sampling())
                    for _ in range(10)
                )
            )
        finally:
            await eval_client.close()
    assert [r.message.content for r in responses] == ["ok"] * 10
    assert provider.peak == 2 and provider.rejected == 0
    (state,) = gate_module.gate_states(tmp_path / "gates")
    assert (state["model"], state["admitted"]) == (MODEL, 10)


CHILD = textwrap.dedent(
    f"""
    import asyncio, sys
    from verifiers.v1.configs.client import RequestGateConfig
    from verifiers.v1.request_gate import GatedAsyncClient

    async def main(url, gates, requests):
        gate = RequestGateConfig(max_concurrent=2, dir=gates)
        async with GatedAsyncClient(gate) as http:
            responses = await asyncio.gather(*(
                http.post(url + "/chat/completions", json={{"model": "{MODEL}", "messages": []}},
                          headers={{"Authorization": "Bearer k"}})
                for _ in range(requests)
            ))
        assert all(r.status_code == 200 for r in responses), [r.status_code for r in responses]

    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
    """
)


@contextlib.asynccontextmanager
async def children(
    provider: FakeProvider, gates: Path, count: int, requests: int
) -> AsyncIterator[list[asyncio.subprocess.Process]]:
    """`count` processes each sending `requests` through the gate (the same client as
    `build_async_httpx` returns, without importing every model client); killed on exit,
    their stderr printed if the test fails."""
    procs = [
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            CHILD,
            provider.url,
            str(gates),
            str(requests),
            stderr=subprocess.PIPE,
        )
        for _ in range(count)
    ]
    try:
        yield procs
    except BaseException:
        for proc in procs:
            if proc.returncode is None:
                proc.kill()
            assert proc.stderr is not None
            stderr = (await proc.stderr.read()).decode(errors="replace")
            print(f"child {proc.pid} stderr:\n{stderr}")
        raise
    finally:
        for proc in procs:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()


async def finished(procs: list[asyncio.subprocess.Process]) -> list[int]:
    for proc in procs:
        assert proc.stderr is not None
        await proc.stderr.read()  # drained, so a chatty child never blocks
    return [await proc.wait() for proc in procs]


async def test_processes_on_a_host_share_one_budget(tmp_path):
    async with (
        asyncio.timeout(TEST_SECONDS),
        FakeProvider(limit=10, latency=0.1) as provider,
        children(provider, tmp_path / "gates", count=4, requests=8) as procs,
    ):
        assert await finished(procs) == [0, 0, 0, 0]
    assert provider.served == 32 and provider.peak == 2 and provider.rejected == 0


async def test_a_resize_reaches_running_processes(tmp_path):
    """Two processes started with a cap of 2 share a raise to 4 made while they run."""
    gates = tmp_path / "gates"
    async with (
        asyncio.timeout(TEST_SECONDS),
        FakeProvider(limit=10, latency=0.1) as provider,
        children(provider, gates, count=2, requests=30) as procs,
    ):
        while provider.served < 4:
            await asyncio.sleep(0.02)
        assert provider.peak == 2
        (resized,) = await asyncio.to_thread(gate_module.resize, gates, 4)
        assert resized["max_concurrent"] == 4
        assert await finished(procs) == [0, 0]
    assert provider.served == 60 and provider.peak == 4 and provider.rejected == 0


async def test_vf_gate_shows_and_resizes_live_gates(tmp_path, capsys):
    gates = tmp_path / "gates"
    async with (
        asyncio.timeout(TEST_SECONDS),
        FakeProvider(limit=10, latency=0.1) as provider,
        children(provider, gates, count=2, requests=30) as procs,
    ):
        while provider.served < 4:
            await asyncio.sleep(0.02)
        await asyncio.to_thread(gate_cli.main, ["--dir", str(gates)])
        table = capsys.readouterr().out.splitlines()
        assert table[0].split() == [*gate_cli.COLUMNS.values(), "updated"]
        assert table[1].split()[:2] == [MODEL, "2"]
        resize = ["--dir", str(gates), "--resize", "3", "--model", MODEL]
        await asyncio.to_thread(gate_cli.main, [*resize, "--output", "json"])
        (resized,) = json.loads(capsys.readouterr().out)
        assert resized["max_concurrent"] == 3
        assert await finished(procs) == [0, 0]
    assert provider.peak == 3


def test_vf_gate_says_when_there_is_no_live_gate(tmp_path, capsys):
    gates = tmp_path / "gates"
    gates.mkdir(mode=0o700)
    gate_cli.main(["--dir", str(gates)])
    assert "no request gates" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no live request gate"):
        gate_cli.main(["--dir", str(gates), "--resize", "4"])


def test_gates_are_keyed_by_endpoint_credential_and_model():
    url = httpx.URL("https://api.example/v1/chat/completions")
    other = httpx.URL("https://other.example/v1/chat/completions")
    key = gate_key(url, "Bearer a", MODEL)
    assert key == gate_key(httpx.URL("https://api.example/v1/messages"), "a", MODEL)
    keys = {
        key,
        gate_key(other, "Bearer a", MODEL),
        gate_key(url, "Bearer b", MODEL),
        gate_key(url, "Bearer a", "x"),
    }
    assert len(keys) == 4


def test_a_stream_collected_unclosed_inside_the_gate_does_not_deadlock(
    tmp_path, monkeypatch
):
    """An unclosed streamed response is freed by the cyclic garbage collector, which can
    run inside the gate's own code, as it sends an admission; its finalizer's release
    must not wait on anything that code holds. The requests run in a thread so a
    deadlock fails the test, not the run."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(b"{}"))

    collect = False
    real_dumps = json.dumps

    def dumps(message, **kwargs):  # the collector firing at an allocation in a send
        if collect:
            gc.collect()
        return real_dumps(message, **kwargs)

    monkeypatch.setattr(
        gate_module, "json", SimpleNamespace(loads=json.loads, dumps=dumps)
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
    assert not thread.is_alive(), "the gate deadlocked on its own send"
    assert statuses == [200], "the collected stream's permit was returned"
