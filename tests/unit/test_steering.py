"""Steering a running agent: `Rollout.steer` through the interception for a harness
with no native mid-turn steering, the per-process registry (`vf.steer`), the env
server's `steer` request, the `message_agent` tool between two agents of an episode,
and the record each steer leaves on its trace.

Deterministic, no API key: a scripted OpenAI-compatible endpoint answers from the
conversation alone (and can hold a request until the test releases it), an in-process
harness drives it through the real interception server on the subprocess runtime.
"""

import asyncio
import contextlib
import json
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
from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.chat import ChatDialect, message_to_wire
from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.harnesses.utils.mcp import mcp_client
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.serve.client import EnvClient
from verifiers.v1.serve.delta import SCALAR_FIELDS, dump
from verifiers.v1.serve.pool import EnvServerPool
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import SteerResponse
from verifiers.v1.session import RolloutSession
from verifiers.v1.trace import TraceTask
from verifiers.v1.types import (
    AssistantMessage,
    Request,
    ToolCall,
    ToolMessage,
    UserMessage,
)

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
    """An in-process tool loop with no native steering: model calls go through the
    interception endpoint, the `write` tool writes into the runtime."""

    SUPPORTS_MCP = False
    SUPPORTS_TOOL_INTERCEPTION = False

    async def setup(self, runtime: vf.Runtime) -> None:
        return None

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
        return web.json_response(reply(body["messages"]))

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


# -- interception delivery ----------------------------------------------------------


async def test_run_is_steerable_through_the_interception(model):
    trace, receipt = await steered_run(model, make(model), NOTE)
    assert receipt == vf.SteerReceipt(
        queued=True, trace_id=trace.id, delivery="interception"
    )
    assert trace.ok, trace.errors
    first, second, third = model.requests
    assert NOTE not in contents(first["messages"])
    # It lands after the tool result the harness sent next, as a user message...
    assert roles(second["messages"]) == ["user", "assistant", "tool", "user"]
    assert second["messages"][3]["content"] == NOTE
    # ...and stays in place on later requests, which the harness sends without it.
    assert roles(third["messages"]) == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
    ]
    assert contents(third["messages"]).count(NOTE) == 1
    assert third["messages"][3]["content"] == NOTE
    assert trace.last_reply == f"done: {NOTE}"
    # One linear conversation: the re-inserted message matched its node every time.
    assert trace.num_branches == 1
    # The record: who, when, how, and the call it landed before.
    (record,) = trace.steers
    assert record.text == NOTE and record.sender is None
    assert record.delivery == "interception"
    assert record.call == 1
    node = trace.nodes[record.node]
    assert node.message == UserMessage(content=NOTE)
    assert not node.sampled and not any(node.mask)
    assert trace.nodes[trace.calls[1].node].parent == record.node
    assert trace.to_record()["steers"][0]["call"] == 1


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


def test_steer_is_reinserted_until_the_harness_carries_it():
    trace = vf.Trace(
        task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="go")),
        agent=vf.AgentInfo(config=vf.AgentConfig(model="m")),
    )
    session = RolloutSession(
        ctx=ModelContext(model="m", client=EvalClientConfig()), trace=trace
    )
    session.queue_steer("stop", sender="me")
    steer = UserMessage(content="stop")
    user = UserMessage(content="go")
    call = AssistantMessage(
        content=None, tool_calls=[ToolCall(id="1", name="write", arguments="{}")]
    )
    result = ToolMessage(tool_call_id="1", content="ok")
    later = AssistantMessage(content="next")
    inserts, placed = session.place_steers(Request(messages=[user, call, result]))
    assert inserts == [(3, steer)] and len(placed) == 1
    # The harness's next request extends its own conversation: back in the same place.
    request = Request(messages=[user, call, result, later, UserMessage(content="u2")])
    assert session.place_steers(request)[0] == [(3, steer)]
    # Another branch (a subagent's conversation) gets nothing; the steer stays live.
    assert session.place_steers(Request(messages=[UserMessage(content="sub")]))[0] == []
    assert session.place_steers(request)[0] == [(3, steer)]
    # Once the harness's own conversation carries it, it is never inserted again.
    echoed = Request(messages=[user, call, result, steer, later])
    assert session.place_steers(echoed)[0] == []
    assert session.place_steers(request)[0] == []
    assert [record.sender for record in trace.steers] == ["me"]


def test_dialects_insert_user_messages():
    steer = UserMessage(content="stop")
    chat = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "go"},
        ]
    }
    dialect = ChatDialect()
    before = dialect.parse_request(chat)
    dialect.insert_user_messages(chat, before, [(1, steer), (2, steer)])
    assert [m.content for m in dialect.parse_request(chat).messages] == [
        "s",
        "stop",
        "go",
        "stop",
    ]

    anthropic = {
        "system": "s",
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "1", "name": "w", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "1", "content": "ok"},
                    {"type": "text", "text": "reminder"},
                ],
            },
        ],
    }
    dialect = AnthropicDialect()
    before = dialect.parse_request(anthropic)
    assert [m.role for m in before.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    # Between a turn's tool result and its text: the turn is split around it.
    dialect.insert_user_messages(anthropic, before, [(4, steer), (5, steer)])
    after = dialect.parse_request(anthropic)
    assert [m.role for m in after.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "user",
        "user",
    ]
    assert [m.content for m in after.messages[4:]] == ["stop", "reminder", "stop"]

    responses = {
        "instructions": "s",
        "input": [
            {"role": "user", "content": "go"},
            {"type": "reasoning", "summary": []},
            {
                "type": "function_call",
                "call_id": "1",
                "name": "w",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "1", "output": "ok"},
        ],
    }
    dialect = ResponsesDialect()
    before = dialect.parse_request(responses)
    assert [m.role for m in before.messages] == ["system", "user", "assistant", "tool"]
    dialect.insert_user_messages(responses, before, [(4, steer)])
    after = dialect.parse_request(responses)
    assert [m.role for m in after.messages][-1] == "user"
    assert after.messages[-1].content == "stop"
    with pytest.raises(ValueError):
        # A request that is not this body parsed is refused, not guessed at.
        dialect.insert_user_messages(responses, Request(messages=[]), [(0, steer)])


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
        assert receipt == {"outcome": "injected", "delivery": "interception"}
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
        assert receipt.queued and receipt.delivery == "interception"
        segment = await interaction.turn("start")
    assert segment.last_reply == f"done: {NOTE}"
    assert roles(model.requests[0]["messages"]) == ["user", "user"]
    assert interaction.trace.steers[0].call == 0


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
        assert response.queued and response.delivery == "interception"
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
            taskset=vf.TasksetConfig(id="echo-v1"),
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
    assert (record.sender, record.text, record.delivery) == (
        "lead",
        NOTE,
        "interception",
    )
    assert record.call == 1
    assert worker.last_reply == f"done: {NOTE}"
    assert lead.steers == []
    assert lead.agent.config.message_agents is True
    assert steering._by_trace == {}


async def test_native_steering_stays_the_fast_path(model, monkeypatch):
    """A harness that takes steering itself (ACP) gets the message natively; the
    interception adds nothing, and the record says so."""
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
