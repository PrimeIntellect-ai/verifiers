"""`Interaction` segments: a failed segment reports its own partial reply."""

from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import Interaction
from verifiers.v1.graph import MessageNode
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.types import AssistantMessage


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
    run.deployment = SimpleNamespace(run=lambda turn: turn)
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
