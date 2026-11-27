"""`Interaction` segments: a failed segment reports its own partial reply, and an
expired agent deadline stops the rollout as a timeout."""

import asyncio
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.agent import Interaction
from verifiers.v1.clients import EvalClientConfig, ModelContext, ModelRoute
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.errors import HarnessError
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.utils.replay import drop_blank_text_blocks
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.session import RolloutSession
from verifiers.v1.types import AssistantMessage


class FailingSegmentRollout:
    ok = Rollout.ok
    closed = Rollout.closed
    fail = Rollout.fail
    step = Rollout.step
    _turn = Rollout._turn
    _segment_agent_time = Rollout._segment_agent_time
    timeout = Rollout.timeout


def make_trace() -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task",
            data=vf.TaskData(idx=0, prompt=None),
            key="dataset/example-0",
            hash="content-digest",
        ),
    )


async def test_each_sample_keeps_its_tool_result_across_assistant_replays():
    replacements = iter(
        ["First result", "Second result", "Follow-up result", "Third result"]
    )

    def replace_result(request: vf.Request) -> vf.Request | None:
        tool = request.messages[-1]
        if isinstance(tool, vf.ToolMessage) and tool.content == "":
            tool.content = next(replacements)
            return request
        return None

    trace = make_trace()
    trace._replay_view = lambda message, dialect: (
        drop_blank_text_blocks(message) if dialect == "anthropic_messages" else message
    )
    sampled = AssistantMessage(
        content="Ready\n",
        text_block_offsets=[5],
        tool_calls=[vf.ToolCall(id="call", name="lookup", arguments="{}")],
    )
    session = RolloutSession(
        ModelRoute(ModelContext(model="test", client=EvalClientConfig())),
        trace,
        request_interceptors=[replace_result],
        gates_tools=True,
    )
    replay = sampled.model_copy(update={"content": "Ready", "text_block_offsets": None})
    other = replay.model_copy(update={"content": "Another action"})
    prompt = [vf.UserMessage(content="go")]
    dialect = AnthropicDialect()

    def restrict(provider_tool: str):
        body = {
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {"name": "lookup", "input_schema": {"type": "object"}},
                {"name": provider_tool.rsplit("_", 1)[0], "type": provider_tool},
            ],
        }
        native = dialect.parse_request(body)
        filtered, _ = dialect.mediate_external_capabilities(
            body, NetworkPolicyConfig(allow=[])
        )
        return native, dialect.parse_request(filtered)

    def commit(messages, assistant, tools=None):
        turn = graph.prepare_turn(
            trace, messages, tools=tools, dialect="anthropic_messages"
        )
        return turn.commit(
            vf.Response(
                id="", created=0, model="test", message=assistant, finish_reason="stop"
            )
        )

    # Removing a provider tool also adds a notice to the model-facing prompt.
    first_native, filtered = restrict("web_search_20250305")
    second_native, second_filtered = restrict("web_fetch_20250910")
    assert filtered == second_filtered
    assert [tool.name for tool in filtered.tools or []] == ["lookup"]
    assert filtered.messages != prompt
    for assistant, replayed, expected, native in [
        (sampled, replay, "First result", first_native),
        (other, other, "Second result", second_native),
    ]:
        node = commit(filtered.messages, assistant, filtered.tools)
        await session.gate_tool_calls(node, "anthropic_messages", native.messages)
        assert (await session.decide_tool("call"))["action"] == "deny"
        request = vf.Request(
            messages=[
                *prompt,
                replayed,
                vf.ToolMessage(tool_call_id="call", content="Tool unavailable"),
            ],
            tools=native.tools,
        )
        rewritten, _, _ = await session.rewrite_request(
            request, dialect="anthropic_messages"
        )
        assert rewritten.messages[-1].content == expected
        if assistant is sampled:
            first_request = request
            first_path = graph.path_node_ids(trace, node)
        model_history = [*filtered.messages, *rewritten.messages[len(prompt) :]]
        model_receipt = vf.Request(
            messages=[*filtered.messages, replayed, request.messages[-1]],
            tools=filtered.tools,
        )
        restored = session.pin_tool_results(model_receipt, "anthropic_messages")
        assert restored.messages[-1].content == expected
        if native is first_native:
            first_model_receipt = model_receipt
        commit(model_history, AssistantMessage(content="done"), filtered.tools)

    # The next request can change its available tools independently of its history.
    changed_catalog = first_request.model_copy(update={"tools": second_native.tools})
    earlier, _, _ = await session.rewrite_request(
        changed_catalog, dialect="anthropic_messages"
    )
    assert earlier.messages[-1].content == "First result"

    # An unrelated branch adds a tool to the trace-wide catalog. The next gate
    # still knows exactly which sampled ancestors issued its historical calls.
    commit(prompt, AssistantMessage(content="other branch"), [vf.Tool(name="other")])
    follow_up = AssistantMessage(
        tool_calls=[vf.ToolCall(id="next", name="lookup", arguments="{}")]
    )
    node = commit(model_history, follow_up, filtered.tools)
    await session.gate_tool_calls(node, "anthropic_messages", rewritten.messages)
    verdict = await session.decide_tool("next")
    assert verdict["message"]["content"] == "Follow-up result"
    continued = request.model_copy(
        update={
            "messages": [
                *request.messages,
                follow_up,
                vf.ToolMessage(tool_call_id="next", content="Tool unavailable"),
            ]
        }
    )
    final, _, _ = await session.rewrite_request(continued, dialect="anthropic_messages")
    assert [m.content for m in final.messages if isinstance(m, vf.ToolMessage)] == [
        "Second result",
        "Follow-up result",
    ]

    # The third sample reuses the first sample's replay and call ID under a new catalog.
    node = commit(filtered.messages, sampled, filtered.tools)
    await session.gate_tool_calls(node, "anthropic_messages", second_native.messages)
    assert (await session.decide_tool("call"))["message"]["content"] == "Third result"
    with pytest.raises(HarnessError, match="Ambiguous issuing turn"):
        await session.rewrite_request(changed_catalog, dialect="anthropic_messages")
    with pytest.raises(HarnessError, match="Ambiguous issuing turn"):
        await session.rewrite_request(first_request, dialect="anthropic_messages")
    with pytest.raises(HarnessError, match="Ambiguous issuing turn"):
        session.pin_tool_results(
            first_request.model_copy(update={"tools": None}), "anthropic_messages"
        )
    with pytest.raises(HarnessError, match="Ambiguous issuing turn"):
        session.pin_tool_results(first_model_receipt, "anthropic_messages")
    restored = session.pin_tool_results(
        changed_catalog, "anthropic_messages", first_path
    )
    assert restored.messages[-1].content == "First result"


def make_rollout(trace: vf.Trace, session, time_remaining=None):
    run = FailingSegmentRollout()
    run.trace = trace
    run._opened = True
    run._closed = False
    run._failed = False
    run._attempt = 0
    run._failure = None
    run._borrowed_runtime = None
    run.runtime = None
    run._agent_time_remaining = time_remaining
    run._agent_time_spent = 0.0
    run._memory = False
    run._snapshot_timer = None
    run._segment_start = None
    run._deadline = None
    run._hold_started = None
    run._hold_deadline = None
    run._held_time = 0.0
    run._unheld = asyncio.Event()
    run._unheld.set()
    run._timeouts = RolloutTimeouts()
    run._harness_session = session
    run._relay = None
    run.deployment = SimpleNamespace(run=lambda turn: turn)
    run._session = SimpleNamespace(
        request_interceptors=[],
        error=None,
        aborted=None,
        abort_event=asyncio.Event(),
        stopped=False,
    )
    run.deadline_at = None
    return run


@pytest.mark.asyncio
async def test_failed_segment_does_not_reuse_prior_root_reply():
    trace = make_trace()
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

    run = make_rollout(trace, FailingSession())

    segment = await Interaction(run).turn("next")

    assert segment.last_reply == "current partial reply"
    assert trace.root_reply is None
    assert trace.last_reply == "current partial reply"


@pytest.mark.asyncio
async def test_expired_deadline_is_a_timeout_when_the_runtime_reports_an_error():
    """A runtime can turn the deadline's cancellation into its own error, as Prime's SDK
    does for a cancelled request. The rollout still stops on its agent timeout."""
    trace = make_trace()

    class CancelReportingSession:
        async def turn(self, messages):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise RuntimeError("request was cancelled") from None

    run = make_rollout(trace, CancelReportingSession(), time_remaining=0.05)

    await Interaction(run).turn("next")

    assert trace.is_timeout and not trace.errors
    assert trace.stop_condition == "agent_timeout"


@pytest.mark.parametrize("outcome", ["keeps waiting", "dies"])
async def test_request_hook_failure_fails_the_rollout(outcome):
    """A request hook's failure is the task's verdict: it ends the segment at once and
    fails the rollout, whatever the harness does meanwhile."""
    trace = make_trace()
    error = vf.ProviderError("hook refused the turn", status_code=503)
    cancelled = []

    class WaitingSession:
        async def turn(self, messages):
            run._session.aborted = error
            run._session.abort_event.set()
            try:
                if outcome == "keeps waiting":
                    await asyncio.sleep(60)
                else:
                    raise RuntimeError("harness died")
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

    run = make_rollout(trace, WaitingSession(), time_remaining=5)

    await asyncio.wait_for(Interaction(run).turn("next"), timeout=1)

    assert [(e.type, e.stage) for e in trace.errors] == [("ProviderError", "agent")]
    assert not trace.is_timeout
    assert cancelled == ([True] if outcome == "keeps waiting" else [])


async def test_task_after_turn_drives_follow_up_segments_and_records_errors(
    monkeypatch,
):
    """`Task.after_turn` runs follow-up segments inside `Agent.run` until it returns
    None; an exception records a `TaskError` on the trace and still closes cleanly."""
    from verifiers.v1 import agent as agent_mod

    seen_steps: list = []
    closed = []

    class FakeRollout:
        ok = Rollout.ok
        fail = Rollout.fail
        timeout = Rollout.timeout

        def __init__(self, *, task, **_kwargs):
            self.task = task
            self.trace = make_trace()
            self.runtime = object()
            self.deployment = None
            self._failed = False
            self._failure = None
            self._borrowed_runtime = None
            self._attempt = 1
            self._agent_time_remaining = None
            self.rewound_snapshots = ()
            self.deleted_snapshots = set()

        async def open(self) -> bool:
            return True

        async def step(self, messages=None) -> bool:
            seen_steps.append(messages)
            return True

        async def close(self) -> vf.Trace:
            self.trace.ok = not self._failed
            closed.append(self.trace)
            return self.trace

        async def abort(self) -> None:
            raise AssertionError("after_turn failure must close, not abort")

    monkeypatch.setattr(agent_mod, "Rollout", FakeRollout)
    agent = vf.Agent(vf.AgentConfig(model="m"))

    class RepairTask(vf.Task):
        async def after_turn(self, trace, runtime):
            count = trace.info.get("checks", 0) + 1
            trace.info["checks"] = count
            return "fix the file" if count == 1 else None

    trace = await agent.run(RepairTask(vf.TaskData(idx=0, prompt="go")))
    assert trace.ok and trace.info["checks"] == 2
    assert seen_steps == [None, [vf.UserMessage(content="fix the file")]]

    class RuntimeOnlyTask(vf.Task):
        async def after_turn(self, runtime):
            assert runtime is not None

    assert (await agent.run(RuntimeOnlyTask(vf.TaskData(idx=0, prompt="go")))).ok

    class FailingCheckTask(vf.Task):
        async def after_turn(self, trace, runtime):
            trace.info["report"] = "bad"
            raise ValueError("invalid report")

    failed = await agent.run(FailingCheckTask(vf.TaskData(idx=0, prompt="go")))
    assert not failed.ok and failed.info["report"] == "bad"
    assert [(e.type, e.stage) for e in failed.errors] == [("TaskError", "agent")]

    class BadReturnTask(vf.Task):
        async def after_turn(self, trace, runtime):
            return False  # type: ignore[return-value]

    bad_return = await agent.run(BadReturnTask(vf.TaskData(idx=0, prompt="go")))
    assert not bad_return.ok
    assert [(e.type, e.stage) for e in bad_return.errors] == [("TaskError", "agent")]

    class SlowCheckTask(vf.Task):
        async def after_turn(self, runtime):
            await asyncio.sleep(10)

    class TimedRollout(FakeRollout):
        def __init__(self, *, task, **kwargs):
            super().__init__(task=task, **kwargs)
            self._agent_time_remaining = 0.02

    monkeypatch.setattr(agent_mod, "Rollout", TimedRollout)
    timed_out = await agent.run(SlowCheckTask(vf.TaskData(idx=0, prompt="go")))
    assert timed_out.stop_condition == "agent_timeout" and not timed_out.errors
    assert len(closed) == 5
