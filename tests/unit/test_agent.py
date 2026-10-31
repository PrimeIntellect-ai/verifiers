"""`Interaction` segments: a failed segment reports its own partial reply, an expired
agent deadline stops the rollout as a timeout, and a lost workspace ends it as a sample
unless its provider lost it."""

import asyncio
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import Interaction
from verifiers.v1.graph import MessageNode
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.runtimes import RuntimeLoss
from verifiers.v1.types import AssistantMessage


class FailingSegmentRollout:
    ok = Rollout.ok
    closed = Rollout.closed
    fail = Rollout.fail
    step = Rollout.step
    timeout = Rollout.timeout
    _workspace_lost = Rollout._workspace_lost
    _workspace_gone = Rollout._workspace_gone


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


def make_rollout(trace: vf.Trace, session, time_remaining=None):
    run = FailingSegmentRollout()
    run.trace = trace
    run._opened = True
    run._closed = False
    run._failed = False
    run._failure = None
    run._borrowed_runtime = None
    run.runtime = None
    run._agent_time_remaining = time_remaining
    run._timeouts = RolloutTimeouts()
    run._harness_session = session
    run.deployment = SimpleNamespace(run=lambda turn: turn, execution_backend=None)
    run._scoring_policy = vf.ScoringConfig()
    run._lost = None
    run._session = SimpleNamespace(
        request_interceptors=[],
        error=None,
        stopped=False,
        release=lambda: None,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("infra", [False, True])
async def test_lost_workspace_is_a_sample_unless_its_provider_lost_it(infra):
    """With `skip_on_workspace_loss`, a harness that dies with its workspace ends the
    rollout as a sample with no rewards; a loss the provider attributes to its own
    infrastructure still fails it."""
    trace = make_trace()
    loss = RuntimeLoss("HOST_OOM" if infra else "sandbox_terminated", infra=infra)

    class DyingSession:
        async def turn(self, messages):
            raise RuntimeError("process stream RPC failed: the sandbox was terminated")

    class LostBackend:
        async def lost(self):
            return loss

    run = make_rollout(trace, DyingSession())
    run._scoring_policy = vf.ScoringConfig(skip_on_workspace_loss=True)
    run.deployment.execution_backend = LostBackend()

    await Interaction(run).turn("next")

    assert trace.info["workspace_loss"] == {"cause": loss.cause}
    if infra:
        assert run._failed and run._lost is None
        assert trace.stop_condition != "workspace_lost"
    else:
        assert not run._failed and run._lost is loss
        assert trace.stop_condition == "workspace_lost" and not trace.errors
