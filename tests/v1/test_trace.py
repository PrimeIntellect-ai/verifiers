"""Trace construction + serialization round-trip: a dumped trace re-validates with plain pydantic
(derived values — reward/is_truncated/error/duration — are properties, not serialized, so they just
recompute on load), transient `state` never crosses the wire, and the permissive `WireTrace` loads a
dump without importing the originating taskset."""

import json
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import Interaction
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.rlm.harness import (
    RLM_SESSION_METADATA_KEY,
    RLMHarness,
    RLMHarnessConfig,
)
from verifiers.v1.lineage import (
    ACP_LINEAGE_HEADERS,
    ACP_LINEAGE_METADATA_KEY,
    extract_lineage_request_id,
)
from verifiers.v1.rollout import Rollout, RolloutTimeouts
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


def _lineage_manifest(
    root_session_id: str, root_status: str = "completed"
) -> vf.LineageManifest:
    return vf.LineageManifest(
        sessions=[
            vf.LineageSession(
                session_id=root_session_id,
                depth=0,
                initial_context_id="ctx-root",
                status=root_status,
            ),
            vf.LineageSession(
                session_id="child",
                parent_session_id=root_session_id,
                depth=1,
                initial_context_id="ctx-child",
                spawned_by_request_id="root-turn",
                status="completed",
            ),
        ],
        contexts=[
            vf.LineageContext(
                context_id="ctx-root", session_id=root_session_id, transition="root"
            ),
            vf.LineageContext(
                context_id="ctx-child", session_id="child", transition="spawn"
            ),
            vf.LineageContext(
                context_id="ctx-root-2",
                session_id=root_session_id,
                previous_context_id="ctx-root",
                transition="compact",
                compaction_id="compact-1",
            ),
        ],
        compactions=[
            vf.LineageCompaction(
                compaction_id="compact-1",
                session_id=root_session_id,
                source_context_id="ctx-root",
                target_context_id="ctx-root-2",
                summary_request_id="root-compact",
                status="completed",
            )
        ],
        requests=[
            vf.LineageRequest(
                request_id="root-turn",
                session_id=root_session_id,
                context_id="ctx-root",
                kind="turn",
            ),
            vf.LineageRequest(
                request_id="child-turn",
                session_id="child",
                context_id="ctx-child",
                kind="turn",
            ),
            vf.LineageRequest(
                request_id="root-compact",
                session_id=root_session_id,
                context_id="ctx-root",
                kind="compaction",
                compaction_id="compact-1",
            ),
            vf.LineageRequest(
                request_id="root-after",
                session_id=root_session_id,
                context_id="ctx-root-2",
                kind="turn",
                compaction_id="compact-1",
            ),
        ],
    )


def test_exact_lineage_joins_interleaved_calls_and_round_trips():
    """Recursive sessions join by exact IDs, not call adjacency or graph shape."""
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
            lineage_request_id="root-turn",
        ),
        vf.ModelCall(
            node=3,
            lineage_request_id="child-turn",
        ),
        vf.ModelCall(
            node=5,
            lineage_request_id="root-compact",
        ),
        vf.ModelCall(
            node=7,
            lineage_request_id="root-after",
        ),
    ]

    manifest = _lineage_manifest(tr.id)
    manifest.sessions.append(
        vf.LineageSession(
            session_id="idle-child",
            parent_session_id=tr.id,
            depth=1,
            initial_context_id="ctx-idle-child",
            spawned_by_request_id="root-turn",
            status="cancelled",
        )
    )
    manifest.contexts.append(
        vf.LineageContext(
            context_id="ctx-idle-child",
            session_id="idle-child",
            transition="spawn",
        )
    )
    tr.reconcile_lineage(vf.LineageManifest.model_validate(manifest.model_dump()))
    assert tr.lineage is not None
    requests = {request.request_id: request for request in tr.lineage.requests}
    assert [requests[call.lineage_request_id].session_id for call in tr.calls] == [
        tr.id,
        "child",
        tr.id,
        tr.id,
    ]

    restored = vf.WireTrace.model_validate_json(tr.model_dump_json())
    assert restored.lineage == tr.lineage
    assert [call.lineage_request_id for call in restored.calls] == [
        call.lineage_request_id for call in tr.calls
    ]
    assert restored.lineage.sessions[-1].session_id == "idle-child"

    # The base ACP layer consumes the generic manifest before the harness's own metadata.
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))
    turn_metadata = {
        ACP_LINEAGE_METADATA_KEY: _lineage_manifest(
            restored.id, root_status="running"
        ).model_dump(mode="json"),
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
    assert restored.lineage.sessions[0].status == "running"

    # session/close carries the terminal snapshot; cumulative metrics overwrite by key.
    close_metadata = {
        ACP_LINEAGE_METADATA_KEY: _lineage_manifest(restored.id).model_dump(
            mode="json"
        ),
        RLM_SESSION_METADATA_KEY: {
            "session_id": restored.id,
            "metrics": {"turns": 4},
        },
    }
    harness._consume_protocol_metadata(restored, close_metadata)
    harness.acp_close_result(restored, close_metadata)
    assert restored.metrics["turns"] == 4
    assert restored.lineage.sessions[0].status == "completed"

    # A failed provider exchange and its SDK retry share one logical request ID.
    restored.calls.append(
        vf.ModelCall(
            lineage_request_id=restored.calls[0].lineage_request_id,
            error=vf.Error(type="E", message="x"),
        )
    )
    restored.reconcile_lineage(_lineage_manifest(restored.id))


def test_lineage_request_id_is_validated_and_stripped():
    headers = {
        "Authorization": "Bearer local",
        "Idempotency-Key": "provider-key",
        "X-ACP-Lineage-Request-ID": "request-1",
        "OpenAI-Beta": "feature",
    }
    request_id, forwarded = extract_lineage_request_id(headers)
    assert request_id == "request-1"
    assert not ACP_LINEAGE_HEADERS.intersection(map(str.lower, forwarded))
    assert forwarded["Idempotency-Key"] == "provider-key"
    assert forwarded["OpenAI-Beta"] == "feature"

    absent, unchanged = extract_lineage_request_id({"OpenAI-Beta": "feature"})
    assert absent is None and unchanged == {"OpenAI-Beta": "feature"}

    with pytest.raises(ValueError, match="not a valid lineage ID"):
        extract_lineage_request_id({"X-ACP-Lineage-Request-ID": "not/a/valid/id"})


def test_acp_lineage_metadata_is_optional_and_agent_session_ids_are_opaque():
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
    )
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))

    harness._consume_protocol_metadata(trace, {})

    assert trace.lineage is None

    harness._consume_protocol_metadata(
        trace,
        {
            ACP_LINEAGE_METADATA_KEY: _lineage_manifest("agent-owned-root").model_dump(
                mode="json"
            )
        },
    )

    assert trace.lineage is not None
    assert trace.lineage.sessions[0].session_id == "agent-owned-root"


def test_lineage_manifest_enforces_root_and_compaction_consistency():
    manifest = _lineage_manifest("root-session").model_dump(mode="json")
    resumed = next(
        request
        for request in manifest["requests"]
        if request["request_id"] == "root-after"
    )
    resumed.pop("compaction_id")

    with pytest.raises(ValueError, match="missing its context compaction"):
        vf.LineageManifest.model_validate(manifest)

    manifest = _lineage_manifest("root-session").model_dump(mode="json")
    manifest["sessions"].append(
        {
            "session_id": "second-root",
            "depth": 0,
            "initial_context_id": "second-root-context",
            "status": "completed",
        }
    )
    manifest["contexts"].append(
        {
            "context_id": "second-root-context",
            "session_id": "second-root",
            "transition": "root",
        }
    )

    with pytest.raises(ValueError, match="exactly one root session"):
        vf.LineageManifest.model_validate(manifest)

    manifest = _lineage_manifest("root-session").model_dump(mode="json")
    manifest["compactions"][0]["status"] = "failed"
    manifest["contexts"] = [
        context
        for context in manifest["contexts"]
        if context["context_id"] != "ctx-root-2"
    ]
    manifest["requests"] = [
        request
        for request in manifest["requests"]
        if request["request_id"] != "root-after"
    ]
    with pytest.raises(ValueError, match="cannot have a target context"):
        vf.LineageManifest.model_validate(manifest)

    manifest["compactions"][0].pop("target_context_id")
    failed = vf.LineageManifest.model_validate(manifest)
    assert failed.compactions[0].target_context_id is None
