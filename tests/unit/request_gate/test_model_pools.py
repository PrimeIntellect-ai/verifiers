"""Model pools: a session binds to the first deployment, in random order, whose request gate
has room, keeps it, and the trace records which deployment and model served each call.
Against local fake providers that count their peak concurrency and answer 429 above it."""

import asyncio
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest
from aiohttp import web
from openai import AsyncOpenAI

import verifiers.v1 as vf
from verifiers.v1 import model_routing
from verifiers.v1 import request_gate as gate_module
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.configs.model_pool import ModelDeployment, ModelPoolConfig
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.runtimes import ProgramResult

GLM, FLASH = "glm-5.3", "glm-5.3-flash"


class FakeProvider:
    """Chat completions after `latency`; a 429 above `limit` in flight."""

    def __init__(self, limit: int = 100, latency: float = 0.05) -> None:
        self.limit, self.latency = limit, latency
        self.active = self.peak = self.rejected = 0
        self.models: Counter[str] = Counter()
        self.url = ""

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if self.active >= self.limit:
            self.rejected += 1
            return web.json_response({"error": {"message": "busy"}}, status=429)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.latency)
            self.models[body["model"]] += 1
            return web.json_response(
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )
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


def deployment(id: str, model: str, url: str, cap: int, gates: Path) -> dict:
    return {
        "id": id,
        "model": model,
        "client": {
            "type": "eval",
            "base_url": url,
            "api_key_var": "FAKE_KEY",
            "request_gate": {"max_concurrent": cap, "dir": str(gates)},
        },
    }


def pool(*deployments: dict, **kwargs) -> ModelPoolConfig:
    return ModelPoolConfig.model_validate({"deployments": list(deployments), **kwargs})


@pytest.fixture(autouse=True)
def fake_key(monkeypatch) -> None:
    monkeypatch.setenv("FAKE_KEY", "k")


# -- config -----------------------------------------------------------------------------


def test_a_deployment_needs_a_gate_and_its_scope_defaults_to_its_id(tmp_path):
    with pytest.raises(ValueError, match="request_gate"):
        ModelDeployment(id="a", model=GLM, client=EvalClientConfig())
    item = ModelDeployment.model_validate(
        deployment("a", GLM, "http://x/v1", 2, tmp_path)
    )
    assert item.client.request_gate is not None
    assert item.client.request_gate.scope == "a"


def test_a_pool_rejects_duplicate_ids_and_mixed_models_when_asked(tmp_path):
    a = deployment("a", GLM, "http://x/v1", 2, tmp_path)
    with pytest.raises(ValueError, match="unique"):
        pool(a, a)
    flash = deployment("b", FLASH, "http://y/v1", 2, tmp_path)
    assert len(pool(a, flash).deployments) == 2
    with pytest.raises(ValueError, match="several models"):
        pool(a, flash, same_model=True)
    replica = deployment("b", GLM, "http://y/v1", 2, tmp_path)
    assert pool(a, replica, same_model=True).same_model


def test_without_a_pool_nothing_changes():
    config = vf.AgentConfig(model=GLM)
    assert config.model_pool is None
    assert "model_pool" not in config.model_dump(mode="json"), "older readers parse it"


# -- choosing a deployment --------------------------------------------------------------


async def test_sessions_fill_every_deployment_up_to_its_cap_and_no_further(tmp_path):
    gates = tmp_path / "gates"
    deployments = pool(
        deployment("a", GLM, "http://a.example/v1", 1, gates),
        deployment("b", GLM, "http://b.example/v1", 2, gates),
        deployment("c", FLASH, "http://c.example/v1", 3, gates),
    )
    chosen = [await model_routing.choose(deployments, ChatDialect()) for _ in range(6)]
    assert Counter(item.id for item, _ in chosen) == {"a": 1, "b": 2, "c": 3}
    waiting = asyncio.create_task(model_routing.choose(deployments, ChatDialect()))
    await asyncio.sleep(0.3)
    assert not waiting.done(), "every deployment is at its cap"
    b_permit = next(permit for item, permit in chosen if item.id == "b")
    b_permit.close()  # a slot of b frees: the waiting session takes it
    item, permit = await asyncio.wait_for(waiting, 10)
    assert item.id == "b"
    for _, held in chosen:
        held.close()
    permit.close()


async def test_a_session_waiting_to_bind_backs_off(tmp_path, monkeypatch):
    """With every deployment full, the retries slow from 30 ms to about 1 s, so many
    waiting sessions cannot flood the daemon."""
    gates = tmp_path / "gates"
    deployments = pool(deployment("a", GLM, "http://a.example/v1", 1, gates))
    _, held = await model_routing.choose(deployments, ChatDialect())
    waits: list[float] = []

    class Stop(Exception):
        pass

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 10:
            raise Stop

    monkeypatch.setattr(
        model_routing,
        "asyncio",
        SimpleNamespace(sleep=sleep, to_thread=asyncio.to_thread),
    )
    with pytest.raises(Stop):
        await model_routing.choose(deployments, ChatDialect())
    held.close()
    assert waits[0] <= 0.03 and waits[-1] > 0.4, waits
    assert all(wait <= 1.0 for wait in waits)
    assert waits[5] > waits[0] * 4, "it grows"


async def test_the_choice_is_spread_at_random_when_all_have_room(tmp_path):
    gates = tmp_path / "gates"
    deployments = pool(
        deployment("a", GLM, "http://a.example/v1", 100, gates),
        deployment("b", GLM, "http://b.example/v1", 100, gates),
    )
    firsts = Counter()
    for _ in range(40):
        item, permit = await model_routing.choose(deployments, ChatDialect())
        firsts[item.id] += 1
        permit.close()
    assert set(firsts) == {"a", "b"}, "no fixed preference order"


def fake_session(pool_config: ModelPoolConfig, info: dict | None = None):
    """What `request_binding` touches of a `RolloutSession`."""
    first = pool_config.deployments[0]
    return SimpleNamespace(
        ctx=vf.ModelContext(
            model=first.model, client=first.client, model_pool=pool_config
        ),
        trace=SimpleNamespace(
            info=dict(info or {}),
            agent=SimpleNamespace(config=vf.AgentConfig()),
            notify=lambda: None,
        ),
        model_lock=asyncio.Lock(),
        deployment=None,
        client=None,
    )


async def test_a_resumed_session_keeps_its_deployment_while_the_pool_has_it(tmp_path):
    gates = tmp_path / "gates"
    a = deployment("a", GLM, "http://a.example/v1", 1, gates)
    b = deployment("b", FLASH, "http://b.example/v1", 1, gates)
    saved = {"model_binding": {"deployment": "b", "model": FLASH}}
    session = fake_session(pool(a, b), saved)
    async with model_routing.request_binding(session, ChatDialect(), lambda c: c):
        assert gate_module.reserved_permit.get() is None, "no new choice was made"
    assert (session.deployment, session.ctx.model) == ("b", FLASH)
    assert session.trace.agent.config.model == FLASH

    gone = fake_session(pool(a), saved)  # b left the pool: bind afresh
    async with model_routing.request_binding(gone, ChatDialect(), lambda c: c):
        permit = gate_module.reserved_permit.get()
        assert permit is not None and not permit.used
    assert gone.deployment == "a"
    assert gone.trace.info["model_binding"]["deployment"] == "a"
    assert permit.used, "an untaken permit is returned when the request ends"


# -- through the interception server ----------------------------------------------------


class CallsHarness(NullHarness):
    """`calls` chat completions through the interception endpoint, then done."""

    SUPPORTS_MCP = False
    SUPPORTS_TOOL_INTERCEPTION = False
    calls = 3

    async def setup(self, runtime: vf.Runtime) -> None:
        return None

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
    ) -> ProgramResult:
        client = AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0)
        try:
            for _ in range(self.calls):
                await client.chat.completions.create(
                    model="whatever", messages=[{"role": "user", "content": "hi"}]
                )
        finally:
            await client.close()
        return ProgramResult(exit_code=0, stdout="", stderr="")


def pooled_agent(model_pool: ModelPoolConfig, calls: int = 3) -> vf.Agent:
    agent = vf.Agent(
        vf.AgentConfig(
            model_pool=model_pool,
            runtime={"type": "subprocess"},
            harness={"id": "null"},
        )
    )
    agent.harness = CallsHarness(agent.harness.config)
    agent.harness.calls = calls
    return agent


async def test_pooled_sessions_spread_stay_put_and_record_their_deployment(tmp_path):
    gates = tmp_path / "gates"
    async with (
        FakeProvider(limit=1, latency=0.3) as glm,
        FakeProvider(limit=1, latency=0.3) as flash,
    ):
        agent = pooled_agent(
            pool(
                deployment("glm-a", GLM, glm.url, 1, gates),
                deployment("flash-b", FLASH, flash.url, 1, gates),
            )
        )
        tasks = [vf.Task(vf.TaskData(idx=i, prompt="go")) for i in range(4)]
        traces = await asyncio.wait_for(
            asyncio.gather(*(agent.run(task) for task in tasks)), 120
        )
    assert all(trace.ok for trace in traces), [trace.errors for trace in traces]
    served = {"glm-a": GLM, "flash-b": FLASH}
    for trace in traces:
        binding = trace.info["model_binding"]
        assert len(trace.calls) == 3
        assert {call.deployment for call in trace.calls} == {binding["deployment"]}
        assert {call.model for call in trace.calls} == {served[binding["deployment"]]}
        assert trace.agent.config.model == served[binding["deployment"]]
    assert {trace.info["model_binding"]["deployment"] for trace in traces} == set(
        served
    )
    assert glm.peak == 1 and flash.peak == 1, "no deployment over its cap"
    assert glm.rejected == 0 and flash.rejected == 0
    assert set(glm.models) == {GLM} and set(flash.models) == {FLASH}, "models not mixed"


async def test_the_chosen_slot_carries_the_first_request(tmp_path):
    gates = tmp_path / "gates"
    async with FakeProvider() as provider:
        agent = pooled_agent(
            pool(deployment("a", GLM, provider.url, 2, gates)), calls=1
        )
        trace = await agent.run(vf.Task(vf.TaskData(idx=0, prompt="go")))
    assert trace.ok, trace.errors
    (state,) = gate_module.gate_states(gates)
    assert state["admitted"] == 1, "one admission: the choice's slot sent the request"
    assert state["in_flight"] == 0


async def test_an_agent_without_a_pool_records_no_deployment(tmp_path):
    async with FakeProvider() as provider:
        agent = vf.Agent(
            vf.AgentConfig(
                model=GLM,
                client={
                    "type": "eval",
                    "base_url": provider.url,
                    "api_key_var": "FAKE_KEY",
                },
                runtime={"type": "subprocess"},
                harness={"id": "null"},
            )
        )
        agent.harness = CallsHarness(agent.harness.config)
        agent.harness.calls = 1
        trace = await agent.run(vf.Task(vf.TaskData(idx=0, prompt="go")))
    assert trace.ok, trace.errors
    assert "model_binding" not in trace.info
    assert [call.deployment for call in trace.calls] == [None]
    assert "deployment" not in trace.calls[0].model_dump(mode="json")
