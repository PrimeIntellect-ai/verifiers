"""Trace construction + serialization round-trip: a dumped trace re-validates with plain pydantic
(derived values — reward/is_truncated/error/duration — are properties, not serialized, so they just
recompute on load), transient `state` never crosses the wire, and the permissive `WireTrace` loads a
dump without importing the originating taskset."""

import asyncio
import gc
import json
import threading
import time
import warnings
from types import SimpleNamespace

import httpx
import pytest
from aiohttp import web

import verifiers.v1 as vf
from verifiers.v1.agent import Interaction
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects import ChatDialect
from verifiers.v1.graph import MessageNode
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.rlm.harness import (
    RLM_SESSION_METADATA_KEY,
    RLMHarness,
    RLMHarnessConfig,
)
from verifiers.v1.interception.server import InterceptionServer
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.runtimes import ProgramResult, SubprocessConfig
from verifiers.v1.semantic import (
    ACP_EXTENSION_HEADERS,
    ACP_SEMANTIC_EDGES_METADATA_KEY,
    extract_acp_info,
)
from verifiers.v1.session import MAX_PENDING_PROGRESS, RolloutSession
from verifiers.v1.types import AssistantMessage, UserMessage


class MyTask(vf.TaskData):
    answer: str = ""  # a task-specific field WireTaskData must absorb


class MyState(vf.State):
    score: int = 0


class FailingSegmentRollout:
    ok = Rollout.ok
    closed = Rollout.closed
    fail = Rollout.fail
    step = Rollout.step


@pytest.mark.asyncio
async def test_failed_segment_does_not_reuse_prior_root_reply():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=0, prompt=None),
            key="dataset/example-0",
            hash="content-digest",
        ),
    )
    trace.root_reply = "previous reply"

    class FailingSession:
        async def turn(self, messages):
            trace.nodes.append(
                MessageNode(
                    parent=None,
                    message=AssistantMessage(content="current partial reply"),
                    sampled=True,
                )
            )
            raise RuntimeError("segment failed after sampling")

    run = FailingSegmentRollout()
    run.trace = trace
    run._opened = True
    run._closed = False
    run._failed = False
    run._failure = None
    run._borrowed_runtime = None
    run.runtime = None
    run._agent_time_remaining = None
    run._timeouts = RolloutTimeouts()
    run._harness_session = FailingSession()
    run._session = SimpleNamespace(
        request_interceptors=[],
        error=None,
        stopped=False,
    )
    run.deadline_at = None

    segment = await Interaction(run).turn("next")

    assert segment.last_reply == "current partial reply"
    assert trace.root_reply is None
    assert trace.last_reply == "current partial reply"


COMPLETION = {
    "id": "cmpl",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


class ProbeHarness(Harness[HarnessConfig]):
    """An in-process chat loop: `calls` model turns through the interception endpoint."""

    NEEDS_CONTAINER = False
    calls = 3

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        messages = [{"role": "user", "content": data.prompt}]
        headers = {"Authorization": f"Bearer {secret}"}
        async with httpx.AsyncClient(base_url=endpoint, headers=headers) as client:
            for _ in range(self.calls):
                body = {"model": ctx.model, "messages": messages}
                reply = await client.post("/chat/completions", json=body)
                reply.raise_for_status()
                messages.append(reply.json()["choices"][0]["message"])
                messages.append({"role": "user", "content": "again"})
        return ProgramResult(exit_code=0, stdout="", stderr="")


@pytest.fixture
async def upstream(monkeypatch):
    """A loopback provider answering every chat completion with `COMPLETION`."""

    async def complete(request: web.Request) -> web.Response:
        return web.json_response(COMPLETION)

    monkeypatch.setenv("UPSTREAM_API_KEY", "test")
    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield EvalClientConfig(
        base_url=f"http://127.0.0.1:{port}/v1", api_key_var="UPSTREAM_API_KEY"
    )
    await runner.cleanup()


async def test_on_progress_observes_live_trace(upstream):
    agent = vf.make_agent(
        vf.AgentConfig(model="m", client=upstream, runtime=SubprocessConfig())
    )
    agent.harness = ProbeHarness(HarnessConfig(id="probe"))
    seen: list[vf.TraceProgress] = []

    def on_progress(progress: vf.TraceProgress) -> None:
        seen.append(progress)
        raise RuntimeError("consumer bug")  # logged, never the rollout's failure

    task = vf.Task(vf.TaskData(idx=0, prompt="hi"))
    trace = await agent.run(task, on_progress=on_progress)

    assert trace.ok and trace.errors == []
    assert len(trace.calls) == trace.num_turns == ProbeHarness.calls
    await settled(
        lambda: len(seen) >= ProbeHarness.calls
    )  # the last delivery is off-loop
    seen.sort(key=lambda p: p.calls)  # deliveries run concurrently in the executor
    assert [p.calls for p in seen] == [1, 2, 3]
    assert [p.nodes for p in seen] == [
        2,
        4,
        6,
    ]  # each call commits its (user, assistant)
    assert all(p.trace_id == trace.id and p.elapsed_s > 0 for p in seen)
    # Each snapshot carries its own copy of the call it announces, never the live record.
    for progress, call in zip(seen, trace.calls):
        assert progress.last_call == call and progress.last_call is not call
        assert progress.last_call.node == call.node is not None


async def settled(done, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not done():
        assert time.monotonic() < deadline, "progress deliveries did not land"
        await asyncio.sleep(0.01)


def slow_sync(seen: list[vf.TraceProgress]):
    def hook(progress: vf.TraceProgress) -> None:
        time.sleep(0.2)
        seen.append(progress)
        raise RuntimeError("consumer bug")

    return hook


def slow_async(seen: list[vf.TraceProgress]):
    async def hook(progress: vf.TraceProgress) -> None:
        await asyncio.sleep(0.2)
        seen.append(progress)
        raise RuntimeError("consumer bug")

    return hook


@pytest.mark.parametrize("make_hook", [slow_sync, slow_async], ids=["sync", "async"])
async def test_on_progress_runs_off_the_event_loop(make_hook, caplog):
    # `record_call` runs in an exchange's `finally`, on the interception server's request
    # path, before the response is flushed. It only snapshots and schedules: a blocking hook
    # runs in the executor (a coroutine hook as a task), so neither the next `record_call`
    # nor the loop itself waits on it — and a raising hook is logged, never raised.
    seen: list[vf.TraceProgress] = []
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="p")),
    )
    session = RolloutSession(
        ctx=ModelContext(model="m", client=EvalClientConfig(base_url="http://x/v1")),
        trace=trace,
        on_progress=make_hook(seen),
    )
    server, dialect = InterceptionServer(), ChatDialect()
    started = time.monotonic()
    for _ in range(2):
        server.record_call(session, dialect, {"model": "m"}, time.time())
    assert time.monotonic() - started < 0.1
    assert len(trace.calls) == 2 and seen == []

    # Ping the loop while the hooks run: a hook on the loop would stall it for 0.2 s.
    worst, last = 0.0, time.monotonic()
    while len(seen) < 2:
        assert time.monotonic() - started < 5, "progress deliveries did not land"
        await asyncio.sleep(0.005)
        now = time.monotonic()
        worst, last = max(worst, now - last), now
    assert worst < 0.1
    assert sorted(p.calls for p in seen) == [1, 2]
    assert all(p.trace_id == trace.id for p in seen)
    assert caplog.text.count("on_progress hook failed") == 2


def make_session(hook) -> RolloutSession:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="p")),
    )
    return RolloutSession(
        ctx=ModelContext(model="m", client=EvalClientConfig(base_url="http://x/v1")),
        trace=trace,
        on_progress=hook,
    )


def stalled_async(entered: list[int]):
    async def hook(progress: vf.TraceProgress) -> None:
        entered.append(progress.calls)
        await asyncio.Event().wait()

    return hook, lambda: None


def stalled_sync(entered: list[int]):
    gate = threading.Event()

    def hook(progress: vf.TraceProgress) -> None:
        entered.append(progress.calls)
        gate.wait()

    return hook, gate.set


@pytest.mark.parametrize("stall", [stalled_sync, stalled_async], ids=["sync", "async"])
async def test_on_progress_bounds_pending_deliveries(stall, caplog):
    # A hook that never returns must not retain one task and one copied call record per
    # model call for the rollout's life: past MAX_PENDING_PROGRESS pending deliveries new
    # snapshots are dropped (logged once), and `release()` cancels what is still pending —
    # a blocking hook's queued executor work included.
    entered: list[int] = []
    hook, unblock = stall(entered)
    session = make_session(hook)
    server, dialect = InterceptionServer(), ChatDialect()
    try:
        for _ in range(50):
            server.record_call(session, dialect, {"model": "m"}, time.time())
            await asyncio.sleep(0)
        assert len(session.trace.calls) == 50
        assert len(session.progress_tasks) == MAX_PENDING_PROGRESS
        assert session.progress_dropped == 50 - MAX_PENDING_PROGRESS
        assert caplog.text.count("dropping new snapshots") == 1
        await asyncio.sleep(0.05)
        assert sorted(entered) == list(range(1, len(entered) + 1))  # in order
        assert 0 < len(entered) <= MAX_PENDING_PROGRESS

        pending = list(session.progress_tasks)
        session.release()
        await asyncio.gather(*pending, return_exceptions=True)
        assert all(task.cancelled() for task in pending)
        assert session.progress_tasks == set()
        # Sealed: a straggler exchange schedules nothing further.
        server.record_call(session, dialect, {"model": "m"}, time.time())
        assert session.progress_tasks == set() and len(session.trace.calls) == 50
    finally:
        unblock()  # let the executor threads a blocking hook holds go


async def test_on_progress_plain_hook_returning_awaitable_completes():
    # `ProgressHook` admits `def hook(p): return async_cb(p)`. The plain callable runs in
    # the executor, but the coroutine it hands back must be awaited on the loop, not
    # dropped (an unawaited coroutine: the callback never runs, and Python warns).
    seen: list[int] = []

    async def async_cb(progress: vf.TraceProgress) -> None:
        await asyncio.sleep(0)
        seen.append(progress.calls)

    def hook(progress: vf.TraceProgress):
        return async_cb(progress)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session = make_session(hook)
        server, dialect = InterceptionServer(), ChatDialect()
        for _ in range(2):
            server.record_call(session, dialect, {"model": "m"}, time.time())
        await settled(lambda: not session.progress_tasks)
        gc.collect()
    assert sorted(seen) == [1, 2]
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


async def test_on_progress_pending_delivery_cancels_at_close(upstream):
    # The rollout's close releases its session, which cancels a delivery the hook is still
    # holding: the conclusion is the returned trace, and a stalled hook retains nothing.
    agent = vf.make_agent(
        vf.AgentConfig(model="m", client=upstream, runtime=SubprocessConfig())
    )
    agent.harness = ProbeHarness(HarnessConfig(id="probe"))
    seen: list[int] = []
    cancelled = asyncio.Event()

    async def on_progress(progress: vf.TraceProgress) -> None:
        seen.append(progress.calls)
        if progress.calls == ProbeHarness.calls:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    task = vf.Task(vf.TaskData(idx=0, prompt="hi"))
    trace = await agent.run(task, on_progress=on_progress)

    assert trace.ok and trace.errors == []
    assert sorted(seen) == [1, 2, 3]
    await settled(cancelled.is_set)


def test_bare_trace_round_trip():
    # The minimal trace: a base task, no nodes, no extras — dump and back into a plain Trace.
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=3, prompt="hello"),
            key="dataset/example-3",
            hash="content-digest",
        ),
    )
    rt = vf.Trace.model_validate(tr.model_dump())
    assert rt.id == tr.id
    assert rt.task.type == "Task"
    assert rt.task.data.idx == 3 and rt.task.data.prompt == "hello"
    assert rt.task.key == "dataset/example-3" and rt.task.hash == "content-digest"
    assert rt.num_turns == 0 and rt.num_branches == 0
    assert rt.reward == 0.0 and rt.errors == []


def test_custom_task_state_round_trip():
    # Custom data and state round-trip into the same parameterization. Data fields are
    # typed (not just `model_extra`); `state` is runtime-only and never crosses the wire.
    tr = vf.Trace[MyTask, MyState](
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="MyTask", data=MyTask(idx=0, prompt="q", answer="gold")),
        state=MyState(score=7),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a"), sampled=True),
        ],
    )
    tr.record_reward("r", 0.5)
    wire = tr.model_dump()
    assert "state" not in wire  # transient state is excluded from the dump

    rt = vf.Trace[MyTask, MyState].model_validate(wire)
    assert (
        isinstance(rt.task.data, MyTask) and rt.task.data.answer == "gold"
    )  # typed custom field
    assert rt.task.type == "MyTask"  # the producing class's name survives the wire
    assert rt.num_turns == 1 and rt.num_branches == 1
    assert rt.reward == 0.5  # property recomputed from `rewards`


def test_wire_trace_round_trip():
    # Two leaves off one root → 2 branches (a compaction-shaped trace), so the round-trip has to
    # carry node `parent` links for `num_branches` to survive.
    tr = vf.Trace[MyTask, vf.State](
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="MyTask", data=MyTask(idx=0, prompt="q", answer="a")),
        tools=[vf.Tool(name="echo", description="", parameters={"type": "object"})],
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(parent=0, message=AssistantMessage(content="a1"), sampled=True),
            MessageNode(parent=0, message=AssistantMessage(content="a2"), sampled=True),
        ],
    )
    tr.record_reward("r", 1.0)
    tr.rewards.setdefault("solved", None)  # seeded: expected but never scored
    tr.metrics.setdefault("acc", None)
    tr.info = {"build": "ok"}
    tr.root_reply = "root answer"
    tr.stop("done")

    # the dump is plain pydantic — derived values are properties, so they're not serialized
    data = json.loads(tr.model_dump_json(exclude_none=True))
    assert "reward" not in data and "is_truncated" not in data
    # exclude_none drops None FIELDS, not None dict values — unscored seeds survive
    assert data["rewards"]["solved"] is None and data["metrics"]["acc"] is None

    rt = vf.WireTrace.model_validate(data)
    assert rt.num_branches == tr.num_branches == 2  # branch topology survived
    assert rt.num_turns == tr.num_turns == 2
    assert rt.reward == 1.0  # property recomputed from `rewards`, seeds contribute 0
    assert rt.rewards["solved"] is None
    assert rt.stop_condition == "done"
    assert rt.info == {"build": "ok"}
    assert rt.root_reply == "root answer"
    assert rt.last_reply == "root answer"
    rt.root_reply = ""
    assert rt.last_reply == ""
    assert (
        rt.tools == tr.tools
    )  # the advertised tools persist (tool-use SFT reads them)
    assert rt.task.data.model_extra == {
        "answer": "a"
    }  # taskset extras preserved on WireTaskData

    # the env-server wire form (a plain model_dump) loads too
    assert vf.WireTrace.model_validate(tr.model_dump()).num_branches == 2


def _semantic_edge_set() -> vf.SemanticEdgeSet:
    return vf.SemanticEdgeSet(
        edges=[
            vf.SemanticEdge(
                source_request_id="root-turn",
                target_request_id="root-compact",
                type="continuation",
            ),
            vf.SemanticEdge(
                source_request_id="root-turn",
                target_request_id="child-turn",
                type="subagent_call",
            ),
            vf.SemanticEdge(
                source_request_id="child-turn",
                target_request_id="root-after",
                type="subagent_return",
            ),
            vf.SemanticEdge(
                source_request_id="root-compact",
                target_request_id="root-after",
                type="compaction",
            ),
            vf.SemanticEdge(
                source_request_id="root-turn",
                target_request_id="root-after",
                type="critic_review",
            ),
        ],
    )


def test_semantic_edges_resolve_to_message_nodes_and_round_trip():
    """Request edges resolve by exact IDs, not call adjacency or graph shape."""
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="root")),
            MessageNode(
                parent=0, message=AssistantMessage(content="root turn"), sampled=True
            ),
            MessageNode(parent=None, message=UserMessage(content="child")),
            MessageNode(
                parent=2, message=AssistantMessage(content="child turn"), sampled=True
            ),
            MessageNode(parent=None, message=UserMessage(content="summarize")),
            MessageNode(
                parent=4, message=AssistantMessage(content="summary"), sampled=True
            ),
            MessageNode(parent=None, message=UserMessage(content="resume")),
            MessageNode(
                parent=6, message=AssistantMessage(content="done"), sampled=True
            ),
        ],
    )
    tr.calls = [
        vf.ModelCall(
            node=1,
            acp=vf.ACPInfo(request_id="root-turn"),
        ),
        vf.ModelCall(
            node=3,
            acp=vf.ACPInfo(request_id="child-turn"),
        ),
        vf.ModelCall(
            node=5,
            acp=vf.ACPInfo(request_id="root-compact"),
        ),
        vf.ModelCall(
            node=7,
            acp=vf.ACPInfo(request_id="root-after"),
        ),
    ]

    edge_set = _semantic_edge_set()
    tr.add_semantic_edges(vf.SemanticEdgeSet(edges=edge_set.edges[:2]))
    first_semantic_parents = tr.nodes[3].semantic_parents
    tr.add_semantic_edges(vf.SemanticEdgeSet.model_validate(edge_set.model_dump()))
    expected_parents = [
        [],
        [],
        [],
        [vf.ParentLink(node=1, type="subagent_call")],
        [],
        [vf.ParentLink(node=1, type="continuation")],
        [],
        [
            vf.ParentLink(node=3, type="subagent_return"),
            vf.ParentLink(node=5, type="compaction"),
            vf.ParentLink(node=1, type="critic_review"),
        ],
    ]
    assert [node.semantic_parents for node in tr.nodes] == expected_parents
    assert tr.nodes[3].semantic_parents is first_semantic_parents

    restored = vf.WireTrace.model_validate_json(tr.model_dump_json())
    assert [node.semantic_parents for node in restored.nodes] == expected_parents
    assert [call.acp for call in restored.calls] == [call.acp for call in tr.calls]

    # The base ACP layer resolves the generic edge set before harness-owned metadata.
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))
    turn_metadata = {
        ACP_SEMANTIC_EDGES_METADATA_KEY: _semantic_edge_set().model_dump(mode="json"),
        RLM_SESSION_METADATA_KEY: {
            "session_id": restored.id,
            "metrics": {"turns": 4},
        },
    }
    harness._consume_protocol_metadata(restored, turn_metadata)
    harness.acp_turn_result(
        restored, vf.ACPTurn(reply="done", response_metadata=turn_metadata)
    )
    assert restored.metrics["turns"] == 4
    assert [node.semantic_parents for node in restored.nodes] == expected_parents

    # session/close may publish the same cumulative edge set again.
    close_metadata = {
        ACP_SEMANTIC_EDGES_METADATA_KEY: _semantic_edge_set().model_dump(mode="json"),
        RLM_SESSION_METADATA_KEY: {
            "session_id": restored.id,
            "metrics": {"turns": 4},
        },
    }
    harness._consume_protocol_metadata(restored, close_metadata)
    harness.acp_close_result(restored, close_metadata)
    assert restored.metrics["turns"] == 4
    assert [node.semantic_parents for node in restored.nodes] == expected_parents

    # A failed provider exchange and its SDK retry share one logical request ID.
    restored.calls.append(
        vf.ModelCall(
            acp=restored.calls[0].acp,
            error=vf.Error(type="E", message="x"),
        )
    )
    restored.add_semantic_edges(_semantic_edge_set())
    assert [node.semantic_parents for node in restored.nodes] == expected_parents


def test_semantic_edge_uses_last_committed_retry_node():
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="root")),
            MessageNode(
                parent=0, message=AssistantMessage(content="attempt 1"), sampled=True
            ),
            MessageNode(
                parent=0, message=AssistantMessage(content="attempt 2"), sampled=True
            ),
            MessageNode(
                parent=None, message=AssistantMessage(content="next"), sampled=True
            ),
        ],
        calls=[
            vf.ModelCall(node=1, acp=vf.ACPInfo(request_id="retried")),
            vf.ModelCall(node=2, acp=vf.ACPInfo(request_id="retried")),
            vf.ModelCall(node=3, acp=vf.ACPInfo(request_id="next")),
        ],
    )

    tr.add_semantic_edges(
        vf.SemanticEdgeSet(
            edges=[
                vf.SemanticEdge(
                    source_request_id="retried",
                    target_request_id="next",
                    type="continuation",
                )
            ]
        )
    )

    assert tr.nodes[3].semantic_parents == [vf.ParentLink(node=2, type="continuation")]


def test_semantic_edge_cycle_is_rejected_without_partial_mutation():
    tr = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="start")),
            MessageNode(
                parent=0, message=AssistantMessage(content="first"), sampled=True
            ),
            MessageNode(parent=1, message=UserMessage(content="continue")),
            MessageNode(
                parent=2, message=AssistantMessage(content="second"), sampled=True
            ),
        ],
        calls=[
            vf.ModelCall(node=1, acp=vf.ACPInfo(request_id="first")),
            vf.ModelCall(node=3, acp=vf.ACPInfo(request_id="second")),
        ],
    )

    with pytest.raises(ValueError, match="cycle in the message graph"):
        tr.add_semantic_edges(
            vf.SemanticEdgeSet(
                edges=[
                    vf.SemanticEdge(
                        source_request_id="second",
                        target_request_id="first",
                        type="custom",
                    )
                ]
            )
        )

    assert all(not node.semantic_parents for node in tr.nodes)


def test_acp_info_is_validated_and_stripped():
    headers = {
        "Authorization": "Bearer local",
        "Idempotency-Key": "provider-key",
        "X-ACP-Model-Request-ID": "request-1",
        "OpenAI-Beta": "feature",
    }
    acp, forwarded = extract_acp_info(headers)
    assert acp == vf.ACPInfo(request_id="request-1")
    assert not ACP_EXTENSION_HEADERS.intersection(map(str.lower, forwarded))
    assert forwarded["Idempotency-Key"] == "provider-key"
    assert forwarded["OpenAI-Beta"] == "feature"

    absent, unchanged = extract_acp_info({"OpenAI-Beta": "feature"})
    assert absent is None and unchanged == {"OpenAI-Beta": "feature"}

    with pytest.raises(ValueError, match="not a valid ACP request ID"):
        extract_acp_info({"X-ACP-Model-Request-ID": "not/a/valid/id"})


def test_acp_semantic_edge_metadata_is_optional():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
    )
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))

    harness._consume_protocol_metadata(trace, {})

    assert all(not node.semantic_parents for node in trace.nodes)

    harness._consume_protocol_metadata(
        trace, {ACP_SEMANTIC_EDGES_METADATA_KEY: {"edges": []}}
    )

    assert all(not node.semantic_parents for node in trace.nodes)


def test_acp_derives_compaction_attempt_branch_trainability():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="work")),
            MessageNode(
                parent=0,
                message=AssistantMessage(content="working"),
                sampled=True,
                token_ids=[1],
                mask=[True],
                logprobs=[-0.1],
            ),
            MessageNode(parent=1, message=UserMessage(content="summarize")),
            MessageNode(
                parent=2,
                message=AssistantMessage(content="bad tool call"),
                sampled=True,
                token_ids=[2, 3],
                mask=[True, True],
                logprobs=[-0.2, -0.3],
            ),
            MessageNode(
                parent=2,
                message=AssistantMessage(content="accepted summary"),
                sampled=True,
                token_ids=[4, 5],
                mask=[True, True],
                logprobs=[-0.4, -0.5],
            ),
            MessageNode(parent=0, message=UserMessage(content="compacted context")),
            MessageNode(
                parent=5,
                message=AssistantMessage(content="answer"),
                sampled=True,
                token_ids=[6],
                mask=[True],
                logprobs=[-0.6],
            ),
        ],
        calls=[
            vf.ModelCall(node=1, acp=vf.ACPInfo(request_id="work")),
            vf.ModelCall(node=3, acp=vf.ACPInfo(request_id="rejected")),
            vf.ModelCall(node=4, acp=vf.ACPInfo(request_id="accepted")),
            vf.ModelCall(node=6, acp=vf.ACPInfo(request_id="resumed")),
        ],
    )
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))
    harness._consume_protocol_metadata(
        trace,
        {
            ACP_SEMANTIC_EDGES_METADATA_KEY: {
                "edges": [
                    {
                        "source_request_id": "work",
                        "target_request_id": "rejected",
                        "type": "compaction_attempt",
                    },
                    {
                        "source_request_id": "work",
                        "target_request_id": "accepted",
                        "type": "compaction_attempt",
                    },
                ]
            },
        },
    )

    attempts = {branch.nodes[-1].message.content: branch for branch in trace.branches}
    assert attempts["bad tool call"].trainable is False
    assert attempts["accepted summary"].trainable is False

    harness._consume_protocol_metadata(
        trace,
        {
            ACP_SEMANTIC_EDGES_METADATA_KEY: {
                "edges": [
                    {
                        "source_request_id": "work",
                        "target_request_id": "rejected",
                        "type": "compaction_attempt",
                    },
                    {
                        "source_request_id": "work",
                        "target_request_id": "accepted",
                        "type": "compaction_attempt",
                    },
                    {
                        "source_request_id": "accepted",
                        "target_request_id": "resumed",
                        "type": "compaction",
                    },
                ]
            },
        },
    )

    assert trace.nodes[3].sampled is True
    assert trace.nodes[3].mask == [True, True]
    assert trace.nodes[4].mask == [True, True]
    assert trace.nodes[6].mask == [True]
    assert trace.nodes[3].semantic_parents == [
        vf.ParentLink(node=1, type="compaction_attempt")
    ]
    assert trace.nodes[4].semantic_parents == [
        vf.ParentLink(node=1, type="compaction_attempt")
    ]
    assert trace.nodes[6].semantic_parents == [vf.ParentLink(node=4, type="compaction")]
    assert trace.num_branches == 3
    branches = {branch.nodes[-1].message.content: branch for branch in trace.branches}
    assert branches["bad tool call"].trainable is False
    assert branches["accepted summary"].trainable is True
    assert branches["answer"].trainable is True
    assert branches["bad tool call"].nodes[-2] is trace.nodes[2]
    assert branches["accepted summary"].nodes[-2] is trace.nodes[2]

    restored = vf.WireTrace.model_validate_json(trace.model_dump_json())
    assert restored.nodes[3].sampled is True
    assert restored.nodes[3].mask == [True, True]
    assert restored.nodes[4].mask == [True, True]
    restored_branches = {
        branch.nodes[-1].message.content: branch for branch in restored.branches
    }
    assert restored_branches["bad tool call"].trainable is False
    assert restored_branches["accepted summary"].trainable is True


def test_semantic_edge_set_rejects_duplicate_self_and_cyclic_edges():
    edge_set = _semantic_edge_set().model_dump(mode="json")
    edge_set["edges"].append(edge_set["edges"][0])
    with pytest.raises(ValueError, match="duplicate semantic edge"):
        vf.SemanticEdgeSet.model_validate(edge_set)

    with pytest.raises(ValueError, match="cannot link a request to itself"):
        vf.SemanticEdgeSet.model_validate(
            {
                "edges": [
                    {
                        "source_request_id": "request-1",
                        "target_request_id": "request-1",
                        "type": "custom",
                    }
                ]
            }
        )

    edge_set = _semantic_edge_set().model_dump(mode="json")
    edge_set["edges"].append(
        {
            "source_request_id": "root-after",
            "target_request_id": "root-turn",
            "type": "custom",
        }
    )
    with pytest.raises(ValueError, match="semantic edge cycle"):
        vf.SemanticEdgeSet.model_validate(edge_set)


def test_semantic_edge_set_accepts_deep_acyclic_chain():
    edge_set = vf.SemanticEdgeSet(
        edges=[
            vf.SemanticEdge(
                source_request_id=f"request-{index}",
                target_request_id=f"request-{index + 1}",
                type="continuation",
            )
            for index in range(2_000)
        ]
    )

    assert len(edge_set.edges) == 2_000
