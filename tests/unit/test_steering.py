"""Steering a running agent: `Rollout.steer` through the harness's inbox for a harness
with no native mid-turn steering, the per-process registry (`vf.steer`), the env
server's `steer` request, the `message_agent` tool between two agents of an episode,
and the record each steer leaves on its trace.

Deterministic, no API key: a scripted OpenAI-compatible endpoint answers from the
conversation alone (and can hold a request until the test releases it). An in-process
harness that reads its inbox like the bundled chat program drives it through the real
interception server on the subprocess runtime; the bash harness tests run that program.
"""

import asyncio
import contextlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import msgpack
import pytest
import zmq
import zmq.asyncio
from aiohttp import web
from openai import AsyncOpenAI

import verifiers.v1 as vf
from verifiers.v1 import steering
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.client import EvalClientConfig
from verifiers.v1.dialects.chat import ChatDialect, message_to_wire
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.harnesses.utils.mcp import mcp_client
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.delta import SCALAR_FIELDS, dump
from verifiers.v1.serve.pool import EnvServerPool
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import SteerResponse
from verifiers.v1.types import UserMessage

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            },
        },
    }
]

NOTE = "also check the file sizes"


def reply(messages: list[dict]) -> dict:
    """The scripted model: write a.txt, then b.txt, then answer the last user turn."""
    tools_seen = sum(message["role"] == "tool" for message in messages)
    if tools_seen < 2:
        name = "ab"[tools_seen] + ".txt"
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{len(messages)}",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": json.dumps({"path": name, "text": name}),
                    },
                }
            ],
        }
        finish = "tool_calls"
    else:
        users = [m["content"] for m in messages if m["role"] == "user"]
        message = {"role": "assistant", "content": f"done: {users[-1]}"}
        finish = "stop"
    return {
        "id": "scripted",
        "object": "chat.completion",
        "created": 0,
        "model": "scripted",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class ScriptHarness(NullHarness):
    """An in-process tool loop with no native steering: before each model call it
    reads its inbox into its conversation, as the bundled chat program does; model
    calls go through the interception endpoint, the `write` tool writes into the
    runtime."""

    SUPPORTS_MCP = False
    SUPPORTS_TOOL_INTERCEPTION = False

    async def setup(self, runtime: vf.Runtime) -> None:
        return None

    async def read_inbox(self, client: AsyncOpenAI, messages: list[dict]) -> None:
        messages += (await client.get("/inbox", cast_to=object))["messages"]

    async def side_calls(self, client: AsyncOpenAI, model: str) -> None:
        """Model calls on a conversation of their own before each main call."""

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
    ) -> ProgramResult:
        system, prompt = self.resolve_prompt(data)
        messages: list[dict] = [{"role": "system", "content": system}] if system else []
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages += [message_to_wire(message) for message in prompt or []]
        client = AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0)
        try:
            while True:
                await self.read_inbox(client, messages)
                await self.side_calls(client, ctx.model)
                completion = await client.chat.completions.create(
                    model=ctx.model, messages=messages, tools=TOOLS
                )
                message = completion.choices[0].message
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    return ProgramResult(exit_code=0, stdout="", stderr="")
                for call in message.tool_calls:
                    args = json.loads(call.function.arguments)
                    await runtime.write(args["path"], args["text"].encode())
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": "ok"}
                    )
        finally:
            await client.close()


@dataclass
class Hold:
    matches: Callable[[dict], bool]
    arrived: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Endpoint:
    url: str
    requests: list[dict] = field(default_factory=list)
    holds: list[Hold] = field(default_factory=list)
    script: Callable[[dict], dict] | None = None
    """Answers a request body; `reply` on its messages when unset."""

    def hold(self, matches: Callable[[dict], bool]) -> Hold:
        """Hold the first request `matches` accepts until `release` is set."""
        hold = Hold(matches)
        self.holds.append(hold)
        return hold


def first_request(body: dict) -> bool:
    return not any(message["role"] == "tool" for message in body["messages"])


@pytest.fixture
async def model(monkeypatch):
    """The scripted endpoint, recording the requests it serves."""
    monkeypatch.setenv("SCRIPTED_KEY", "scripted")
    endpoint = Endpoint("")

    async def handle(request: web.Request) -> web.Response:
        body = await request.json()
        endpoint.requests.append(body)
        for hold in list(endpoint.holds):
            if hold.matches(body):
                endpoint.holds.remove(hold)
                hold.arrived.set()
                await hold.release.wait()
                break
        completion = (
            endpoint.script(body) if endpoint.script else reply(body["messages"])
        )
        if not body.get("stream"):
            return web.json_response(completion)
        return web.Response(
            body=b"".join(ChatDialect().stream_events(completion)),
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    endpoint.url = f"http://127.0.0.1:{port}/v1"
    yield endpoint
    for hold in endpoint.holds:
        hold.release.set()
    await runner.cleanup()


def client_config(model: Endpoint) -> dict:
    return {"type": "eval", "base_url": model.url, "api_key_var": "SCRIPTED_KEY"}


def make(model: Endpoint, harness=ScriptHarness, **config) -> vf.Agent:
    agent = vf.Agent(
        vf.AgentConfig(
            model="scripted",
            client=client_config(model),
            runtime={"type": "subprocess"},
            harness={"id": "null"},
            **config,
        )
    )
    agent.harness = harness(agent.harness.config)
    return agent


def task(prompt: str | None = "start") -> vf.Task:
    return vf.Task(vf.TaskData(idx=0, prompt=prompt))


def roles(messages) -> list[str]:
    return [message["role"] for message in messages]


def contents(messages) -> list:
    return [message.get("content") for message in messages]


async def steered_run(model: Endpoint, agent: vf.Agent, text: str, **steer):
    """Run `agent`, steering it by its trace id (plus `steer` options) while its first
    model request is held upstream; returns the trace and the steer's receipt."""
    hold = model.hold(first_request)
    traces: list[vf.Trace] = []
    run = asyncio.create_task(agent.run(task(), on_trace=traces.append))
    await hold.arrived.wait()
    receipt = await vf.steer(text, trace_id=traces[0].id, **steer)
    hold.release.set()
    return await run, receipt


# -- default path ---------------------------------------------------------------


async def test_default_run_records_no_steers(model):
    trace = await make(model).run(task())
    assert trace.ok, trace.errors
    assert trace.steers == []
    assert "steers" not in trace.to_record()
    assert "steers" not in dump(trace, include=set(SCALAR_FIELDS))
    assert [roles(r["messages"]) for r in model.requests] == [
        ["user"],
        ["user", "assistant", "tool"],
        ["user", "assistant", "tool", "assistant", "tool"],
    ]
    assert trace.agent.config.message_agents is False
    assert trace.id not in steering._by_trace


# -- inbox delivery -----------------------------------------------------------------


async def test_run_is_steerable_through_the_inbox(model):
    trace, receipt = await steered_run(model, make(model), NOTE)
    assert receipt == vf.SteerReceipt(queued=True, trace_id=trace.id, delivery="inbox")
    assert trace.ok, trace.errors
    first, second, third = model.requests
    assert NOTE not in contents(first["messages"])
    # The harness read it before its next call: a user message after the tool result...
    assert roles(second["messages"]) == ["user", "assistant", "tool", "user"]
    assert second["messages"][3]["content"] == NOTE
    # ...that stays where the harness keeps it in its own history.
    assert roles(third["messages"]) == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
    ]
    assert contents(third["messages"]).count(NOTE) == 1
    assert trace.last_reply == f"done: {NOTE}"
    assert trace.num_branches == 1
    assert [m.content for m in trace.messages].count(NOTE) == 1
    # The record: who, when, how, and the call it landed before.
    (record,) = trace.steers
    assert record.text == NOTE and record.sender is None
    assert record.delivery == "inbox"
    assert record.call == 1
    node = trace.nodes[record.node]
    assert node.message == UserMessage(content=NOTE)
    assert not node.sampled and not any(node.mask)
    assert trace.nodes[trace.calls[1].node].parent == record.node
    assert trace.to_record()["steers"][0]["call"] == 1


async def test_queued_steers_arrive_in_order(model):
    agent = make(model)
    hold = model.hold(first_request)
    traces: list[vf.Trace] = []
    run = asyncio.create_task(agent.run(task(), on_trace=traces.append))
    await hold.arrived.wait()
    for text in ("one", "two", "three"):
        assert (await vf.steer(text, trace_id=traces[0].id)).queued
    hold.release.set()
    trace = await run
    assert trace.ok, trace.errors
    second = model.requests[1]["messages"]
    assert roles(second) == ["user", "assistant", "tool", "user", "user", "user"]
    assert contents(second)[3:] == ["one", "two", "three"]
    assert [(s.text, s.call) for s in trace.steers] == [
        ("one", 1),
        ("two", 1),
        ("three", 1),
    ]
    nodes = [s.node for s in trace.steers]
    assert nodes == sorted(nodes) and len(set(nodes)) == 3
    assert trace.last_reply == "done: three"


class IgnoresInbox(ScriptHarness):
    """Reads its inbox and drops what it gets."""

    async def read_inbox(self, client: AsyncOpenAI, messages: list[dict]) -> None:
        await client.get("/inbox", cast_to=object)


async def test_the_interception_never_adds_a_steer_to_a_request(model):
    trace, receipt = await steered_run(model, make(model, IgnoresInbox), NOTE)
    assert receipt.queued and trace.ok, trace.errors
    assert len(model.requests) == 3
    assert all(NOTE not in contents(r["messages"]) for r in model.requests)
    (record,) = trace.steers
    assert (record.call, record.node) == (None, None)
    assert NOTE not in [m.content for m in trace.messages]


class SideBranch(ScriptHarness):
    """Before each main call, one call on a separate conversation (a subagent's)."""

    async def side_calls(self, client: AsyncOpenAI, model: str) -> None:
        await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "sub"}]
        )


async def test_a_steer_lands_on_the_conversation_that_carries_it(model):
    agent = make(model, SideBranch)
    hold = model.hold(lambda body: body["messages"][0]["content"] == "start")
    traces: list[vf.Trace] = []
    run = asyncio.create_task(agent.run(task(), on_trace=traces.append))
    await hold.arrived.wait()
    assert (await vf.steer(NOTE, trace_id=traces[0].id)).queued
    hold.release.set()
    trace = await run
    assert trace.ok, trace.errors
    side = [r for r in model.requests if r["messages"][0]["content"] == "sub"]
    assert side and all(roles(r["messages"]) == ["user"] for r in side)
    (record,) = trace.steers
    # The side call right after the read does not hold it; the main call after it does.
    assert record.call == 3
    assert trace.nodes[trace.calls[3].node].parent == record.node
    side_node = trace.calls[2].node
    assert trace.nodes[side_node].parent is not None
    assert trace.nodes[trace.nodes[side_node].parent].message == UserMessage(
        content="sub"
    )


class NoInbox(ScriptHarness):
    SUPPORTS_INBOX = False


async def test_a_harness_without_steering_refuses_it(model):
    agent = make(model, NoInbox)
    trace, receipt = await steered_run(model, agent, NOTE)
    assert receipt.queued is False and receipt.reason
    assert trace.ok, trace.errors
    assert trace.steers == []
    assert all(NOTE not in contents(r["messages"]) for r in model.requests)
    hold = model.hold(first_request)
    async with agent.interaction(task()) as interaction:
        turn = asyncio.create_task(interaction.turn())
        await hold.arrived.wait()
        with pytest.raises(RuntimeError):
            await interaction.steer(NOTE)
        hold.release.set()
        await turn
    assert interaction.trace.steers == []


async def test_a_closed_run_is_no_longer_steerable(model):
    trace = await make(model).run(task())
    receipt = await vf.steer(NOTE, trace_id=trace.id)
    assert receipt.queued is False
    assert "no running agent" in receipt.reason
    with pytest.raises(ValueError):
        await vf.steer(" ", trace_id=trace.id)


async def test_steer_message_ids_deduplicate(model):
    agent = make(model)
    hold = model.hold(first_request)
    traces: list[vf.Trace] = []
    run = asyncio.create_task(agent.run(task(), on_trace=traces.append))
    await hold.arrived.wait()
    for _ in range(2):
        receipt = await vf.steer(NOTE, trace_id=traces[0].id, message_id="n1")
        assert receipt.queued
    other = await vf.steer("different", trace_id=traces[0].id, message_id="n1")
    assert other.queued is False and "reused" in other.reason
    hold.release.set()
    trace = await run
    assert [record.id for record in trace.steers] == ["n1"]
    assert contents(model.requests[1]["messages"]).count(NOTE) == 1


# -- interactions --------------------------------------------------------------------


async def test_interaction_steers_a_harness_without_native_steering(model):
    agent = make(model)
    hold = model.hold(first_request)
    async with agent.interaction(task()) as interaction:
        # Idle (no turn running): the owner decides whether to wake it.
        assert await interaction.steer("early") == {
            "outcome": "promptRequired",
            "reason": "noRunningTurn",
        }
        turn = asyncio.create_task(interaction.turn())
        await hold.arrived.wait()
        receipt = await interaction.steer(NOTE, message_id="m1")
        assert receipt == {"outcome": "injected", "delivery": "inbox"}
        assert await interaction.steer(NOTE, message_id="m1") == receipt
        hold.release.set()
        segment = await turn
        assert segment.last_reply == f"done: {NOTE}"
    trace = interaction.trace
    assert [(s.text, s.sender, s.call, s.id) for s in trace.steers] == [
        (NOTE, None, 1, "m1")
    ]
    assert trace.id not in steering._by_trace


async def test_registry_steers_an_idle_interaction_on_its_next_turn(model):
    agent = make(model)
    async with agent.interaction(task(None)) as interaction:
        receipt = await vf.steer(NOTE, trace_id=interaction.trace.id, sender="monitor")
        assert receipt.queued and receipt.delivery == "inbox"
        segment = await interaction.turn("start")
    assert segment.last_reply == f"done: {NOTE}"
    assert roles(model.requests[0]["messages"]) == ["user", "user"]
    assert interaction.trace.steers[0].call == 0


# -- the bash harness's chat program -------------------------------------------------


def bash_reply(body: dict) -> dict:
    """The scripted model for the bash program: run one command, then answer the last
    user message; a compaction checkpoint (no tools allowed) gets a summary. Its first
    reply reports a large prompt, which a low compaction threshold acts on."""
    messages = body["messages"]
    users = [m["content"] for m in messages if m["role"] == "user"]
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    if body.get("tool_choice") == "none":
        message, finish = {"role": "assistant", "content": "summary"}, "stop"
    elif len(users) == 1 and not any(m["role"] == "tool" for m in messages):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "true"}),
                    },
                }
            ],
        }
        finish = "tool_calls"
        usage = {"prompt_tokens": 5000, "completion_tokens": 1, "total_tokens": 5001}
    else:
        message, finish = {"role": "assistant", "content": f"done: {users[-1]}"}, "stop"
    return {
        "id": "scripted",
        "object": "chat.completion",
        "created": 0,
        "model": "scripted",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage,
    }


@pytest.fixture
def bash_program(model, tmp_path, monkeypatch):
    """Run the bundled chat program with this interpreter (its dependencies are the
    test environment's) instead of a uv-provisioned one."""
    model.script = bash_reply

    async def prepare(self, script, env=None, *, activate=True):
        path = tmp_path / "chat_program.py"
        path.write_bytes(script.encode() if isinstance(script, str) else script)
        return [sys.executable, str(path)]

    monkeypatch.setattr(Runtime, "prepare_uv_script", prepare)
    return model


def bash_agent(model: Endpoint, **harness) -> vf.Agent:
    return vf.Agent(
        vf.AgentConfig(
            model="scripted",
            client=client_config(model),
            runtime={"type": "subprocess"},
            harness={"id": "bash", "edit": False, "compaction": None, **harness},
        )
    )


def opened(body: dict) -> bool:
    return not any(message["role"] == "tool" for message in body["messages"])


async def bash_steered_run(model: Endpoint, agent: vf.Agent, *texts: str) -> vf.Trace:
    hold = model.hold(opened)
    traces: list[vf.Trace] = []
    run = asyncio.create_task(agent.run(task(), on_trace=traces.append))
    await hold.arrived.wait()
    for text in texts:
        receipt = await vf.steer(text, trace_id=traces[0].id, sender="operator")
        assert receipt.queued and receipt.delivery == "inbox"
    hold.release.set()
    return await run


async def test_bash_program_takes_steers_into_its_history(bash_program):
    model = bash_program
    trace = await bash_steered_run(model, bash_agent(model), "one", "two")
    assert trace.ok, trace.errors
    first, second = model.requests
    assert "one" not in contents(first["messages"])
    # Both arrive, in order, at the next model call, after the command's result.
    assert roles(second["messages"])[-3:] == ["tool", "user", "user"]
    assert contents(second["messages"])[-2:] == ["one", "two"]
    assert trace.last_reply == "done: two"
    conversation = [m.content for m in trace.messages]
    assert conversation.count("one") == 1 and conversation.count("two") == 1
    assert [(s.text, s.call, s.sender) for s in trace.steers] == [
        ("one", 1, "operator"),
        ("two", 1, "operator"),
    ]
    assert all(
        trace.nodes[s.node].message == UserMessage(content=s.text) for s in trace.steers
    )


async def test_bash_program_compacts_without_the_steer_reinserted(bash_program):
    model = bash_program
    agent = bash_agent(model, compaction={"summarize_at_tokens": 1000})
    trace = await bash_steered_run(model, agent, NOTE)
    assert trace.ok, trace.errors
    _, checkpoint, after = model.requests
    # The checkpoint summarizes the conversation as it stood: the steer was not read yet.
    assert checkpoint.get("tool_choice") == "none"
    assert NOTE not in contents(checkpoint["messages"])
    # The rebuilt conversation takes it at the next model call, once.
    assert roles(after["messages"]) == ["system", "user", "user"]
    assert contents(after["messages"])[-1] == NOTE
    assert trace.last_reply == f"done: {NOTE}"
    (record,) = trace.steers
    assert record.call == 2
    assert trace.nodes[trace.calls[2].node].parent == record.node


# -- env server ----------------------------------------------------------------------


class _NoServing:
    @contextlib.asynccontextmanager
    async def serving(self):
        yield


async def test_env_server_steer_request(model):
    """A `steer` request over the env server's zmq protocol reaches the worker's
    registry; `EnvClient.steer` is how a client (an orchestrator) sends it."""
    server = EnvServer.__new__(EnvServer)
    server.taskset_id = "steer"
    server.env = _NoServing()
    server._running = {}
    server._gate = None
    server.ctx = zmq.asyncio.Context()
    server.frontend = server.ctx.socket(zmq.ROUTER)
    server.frontend.setsockopt(zmq.LINGER, 0)
    server.frontend.bind("tcp://127.0.0.1:0")
    server.address = server.frontend.getsockopt_string(zmq.LAST_ENDPOINT)
    serving = asyncio.create_task(server.run())
    client = EnvClient(server.address)
    try:
        agent = make(model)
        hold = model.hold(first_request)
        traces: list[vf.Trace] = []
        run = asyncio.create_task(agent.run(task(), on_trace=traces.append))
        await hold.arrived.wait()
        response = await client.steer(NOTE, trace_id=traces[0].id, sender="operator")
        assert response.queued and response.delivery == "inbox"
        assert response.trace_id == traces[0].id
        missing = await client.steer(NOTE, trace_id="nope")
        assert not missing.queued and "no running agent" in missing.reason
        blank = await client.steer(" ", trace_id=traces[0].id)
        assert not blank.queued and "nonempty" in blank.reason
        hold.release.set()
        trace = await run
        assert trace.steers[0].sender == "operator"
        assert trace.last_reply == f"done: {NOTE}"
    finally:
        await client.close()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)


class _Frontend:
    def __init__(self) -> None:
        self.sent: list[list[bytes]] = []

    async def send_multipart(self, frames, **_) -> None:
        self.sent.append(frames)


async def test_pool_answers_a_steer_from_the_worker_that_queued_it():
    pool = EnvServerPool.__new__(EnvServerPool)
    pool.frontend = _Frontend()
    no = msgpack.packb(SteerResponse(reason="none").model_dump(mode="json"))
    yes = msgpack.packb(SteerResponse(queued=True).model_dump(mode="json"))
    steers = {b"r1": {"client_id": b"c", "waiting": 3, "answered": False}}
    await pool._answer_steer(steers, b"r1", no)
    assert pool.frontend.sent == []
    await pool._answer_steer(steers, b"r1", yes)
    assert pool.frontend.sent == [[b"c", b"r1", b"reply", yes]]
    await pool._answer_steer(steers, b"r1", no)
    assert steers == {} and len(pool.frontend.sent) == 1
    # No worker runs it: the last answer goes back.
    steers = {b"r2": {"client_id": b"c", "waiting": 2, "answered": False}}
    await pool._answer_steer(steers, b"r2", no)
    await pool._answer_steer(steers, b"r2", no)
    assert pool.frontend.sent[-1] == [b"c", b"r2", b"reply", no]


# -- message_agent between two agents of an episode -----------------------------------


class PairConfig(vf.EnvConfig):
    lead: vf.AgentConfig = vf.AgentConfig(message_agents=True)
    worker: vf.AgentConfig = vf.AgentConfig()


class LeadHarness(ScriptHarness):
    """Calls `message_agent` through its MCP server before its own tool loop, once
    the worker is mid-turn."""

    SUPPORTS_MCP = True
    worker_busy: asyncio.Event
    told: list[str]

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
    ) -> ProgramResult:
        await self.worker_busy.wait()
        async with mcp_client({"url": mcp_urls["agents"]}) as client:
            for seat, text in [("worker", NOTE), ("lead", "me"), ("nobody", "x")]:
                result = await client.call_tool(
                    "message_agent", {"seat": seat, "text": text}
                )
                self.told.append(result.content[0].text)
        return await super().launch(
            ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
        )


class PairEnv(vf.Env[PairConfig]):
    hold: Hold
    told: list[str]

    async def run(self, task, agents):
        lead = LeadHarness(agents.lead.harness.config)
        lead.worker_busy, lead.told = self.hold.arrived, self.told
        agents.lead.harness = lead
        agents.worker.harness = ScriptHarness(agents.worker.harness.config)

        async def lead_run():
            try:
                await agents.lead.run(task.with_data(prompt="lead"))
            finally:
                self.hold.release.set()

        await asyncio.gather(
            lead_run(), agents.worker.run(task.with_data(prompt="work"))
        )


async def test_message_agent_steers_a_sibling_seat(model):
    seat = {"runtime": {"type": "subprocess"}, "harness": {"id": "null"}}
    env = PairEnv(
        PairConfig(
            taskset=vf.TasksetConfig(id="echo"),
            max_concurrent_agents=None,
            lead=vf.AgentConfig(**seat, message_agents=True),
            worker=vf.AgentConfig(**seat),
        )
    )
    env.hold = model.hold(
        lambda body: body["messages"][0]["content"] == "work" and first_request(body)
    )
    env.told = []
    ctx = ModelContext(
        model="scripted", client=EvalClientConfig(**client_config(model))
    )
    episode = await env.run_episode(task(), ctx)
    assert episode.ok, (episode.errors, [t.errors for t in episode.traces])
    assert env.told[0] == "delivered to worker"
    assert env.told[1] == "not delivered: that seat is this agent"
    assert env.told[2] == "not delivered: no agent 'nobody' in this episode"
    worker = next(t for t in episode.traces if t.agent.name == "worker")
    lead = next(t for t in episode.traces if t.agent.name == "lead")
    (record,) = worker.steers
    assert (record.sender, record.text, record.delivery) == ("lead", NOTE, "inbox")
    assert record.call == 1
    assert worker.last_reply == f"done: {NOTE}"
    assert lead.steers == []
    assert lead.agent.config.message_agents is True
    assert steering._by_trace == {}


async def test_native_steering_stays_the_fast_path(model, monkeypatch):
    """A harness that takes steering itself (ACP) gets the message natively; its
    inbox stays empty, and the record says so."""
    accepted: list[str] = []

    async def native(self, message, *, message_id=None):
        accepted.append(message)
        return {"outcome": "injected"}

    monkeypatch.setattr(vf.HarnessSession, "steer", native)
    trace, receipt = await steered_run(model, make(model), NOTE, sender="monitor")
    assert receipt.queued and receipt.delivery == "native"
    assert accepted == [NOTE]
    assert all(NOTE not in contents(r["messages"]) for r in model.requests)
    (record,) = trace.steers
    assert (record.delivery, record.sender, record.call, record.node) == (
        "native",
        "monitor",
        0,
        None,
    )


class _DeadDealer:
    """A dead worker's socket: what it sent before it died, then nothing."""

    def __init__(self, *messages: list[bytes]) -> None:
        self.messages = [[zmq.Frame(part) for part in m] for m in messages]
        self.closed = False

    async def recv_multipart(self, flags=0, copy=True):
        if not self.messages:
            raise zmq.Again()
        return self.messages.pop(0)

    def close(self) -> None:
        self.closed = True


class _DeadProcess:
    sentinel = -1
    exitcode = -9

    def join(self, timeout=None) -> None:
        pass


async def test_a_dead_worker_answers_what_it_held(tmp_path):
    """A worker dies holding a run, a steer's share and a reply it sent first: the
    reply is relayed, the run answered as failed and the steer as not queued, and the
    worker leaves the pool."""
    pool = EnvServerPool.__new__(EnvServerPool)
    pool.frontend = _Frontend()
    pool._poller = None
    pool._ipc_dir = str(tmp_path)
    done = msgpack.packb({"success": True})
    dead = {
        "index": 0,
        "active": 2,
        "dealer": _DeadDealer([b"r-done", b"reply", done]),
        "process": _DeadProcess(),
    }
    alive = {"index": 1, "active": 0}
    pool.workers = [dead, alive]
    pending = {
        b"r-done": {"client_id": b"c", "worker": dead},
        b"r-run": {"client_id": b"c", "worker": dead},
    }
    steers = {b"s": {"client_id": b"c", "waiting": 2, "answered": False}}
    steer_parts = {b"s/0": b"s", b"s/1": b"s"}
    closed = await pool._drop_worker(dead, pending, steer_parts, steers)
    assert closed == 2 and pending == {}
    assert pool.workers == [alive] and dead["dealer"].closed
    replies = {bytes(frames[1]): frames for frames in pool.frontend.sent}
    assert bytes(replies[b"r-done"][3]) == done
    lost = msgpack.unpackb(replies[b"r-run"][3])
    assert not lost["success"] and "died" in lost["error"]
    # The steer still waits on the live worker's share, which answers the client.
    assert steer_parts == {b"s/1": b"s"} and steers[b"s"]["waiting"] == 1
    assert b"s" not in replies
    # The last worker dies too: its share answers the steer, and the pool stops.
    alive.update(dealer=_DeadDealer(), process=_DeadProcess())
    with pytest.raises(RuntimeError, match="every env worker died"):
        await pool._drop_worker(alive, pending, steer_parts, steers)
    assert steers == {} and steer_parts == {}
    assert not msgpack.unpackb(pool.frontend.sent[-1][3])["queued"]
