"""`Interaction` segments: a failed segment reports its own partial reply, an
expired agent deadline stops the rollout as a timeout, and a harness dying on a
context overflow truncates it."""

import asyncio
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import Interaction
from verifiers.v1.errors import (
    ContextOverflowError,
    HarnessError,
    InterceptionError,
    ProviderError,
)
from verifiers.v1.graph import MessageNode
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.types import AssistantMessage


class FailingSegmentRollout:
    ok = Rollout.ok
    closed = Rollout.closed
    fail = Rollout.fail
    step = Rollout.step
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
    run.deployment = SimpleNamespace(run=lambda turn: turn)
    run._session = SimpleNamespace(
        request_interceptors=[],
        error=None,
        context_overflow=None,
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


class DyingSession:
    """A harness turn that dies with `error` after `before` runs."""

    def __init__(self, before, error=None):
        self.before = before
        self.error = error or HarnessError("prime-agent turn failed: 400")

    async def turn(self, messages):
        self.before()
        raise self.error


@pytest.mark.asyncio
async def test_harness_dying_on_an_overflow_truncates_the_rollout():
    """A parallel agent's later model turn clears the stashed error before the harness
    dies on the overflow, as a Prime Agent subagent does; the overflow still truncates."""
    trace = make_trace()
    run = make_rollout(trace, None)
    overflow = ContextOverflowError(
        "Prompt length (46590) exceeds maximum context length"
    )

    def overflow_then_sibling_turn():
        run._session.error = run._session.context_overflow = overflow
        run._session.error = None

    run._harness_session = DyingSession(overflow_then_sibling_turn)

    assert not await run.step()
    assert trace.stop_condition == "prompt_too_long" and trace.is_truncated
    assert not trace.errors and not run._failed


@pytest.mark.asyncio
async def test_a_later_model_failure_outranks_an_earlier_overflow():
    trace = make_trace()
    run = make_rollout(trace, None)

    def overflow_then_outage():
        run._session.context_overflow = ContextOverflowError("overflow")
        run._session.error = ProviderError("upstream down", status_code=503)

    run._harness_session = DyingSession(overflow_then_outage)

    assert not await run.step()
    assert trace.stop_condition == "provider_error"
    assert [e.type for e in trace.errors] == ["ProviderError"]


@pytest.mark.asyncio
async def test_an_infra_failure_after_an_overflow_still_fails():
    trace = make_trace()
    run = make_rollout(trace, None)

    def overflow():
        run._session.context_overflow = ContextOverflowError("overflow")

    run._harness_session = DyingSession(overflow, InterceptionError("tunnel dropped"))

    assert not await run.step()
    assert [e.type for e in trace.errors] == ["InterceptionError"]
