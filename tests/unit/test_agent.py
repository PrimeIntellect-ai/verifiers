"""`Interaction` segments: a failed segment reports its own partial reply, and an
expired agent deadline stops the rollout as a timeout. `Agent.with_config` keeps the
agent's standing."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import verifiers.v1 as vf
from verifiers.v1.agent import Interaction, TimeoutConfig, _EpisodeAgent
from verifiers.v1.graph import MessageNode
from verifiers.v1.harnesses.null import NullHarnessConfig
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


def test_with_config_keeps_an_episode_agents_standing():
    """A per-run config (a harness naming a per-run credential, a borrowed box's
    runtime policy) on an env's episode agent: a copy that is still the episode's."""
    completed, gate, warned = [], asyncio.Semaphore(1), set()
    seat = _EpisodeAgent(
        vf.AgentConfig(model="m", runtime={"type": "subprocess"}),
        interception=None,
        name="solver",
        shared_tools={},
        task_cls=vf.Task,
        gate=gate,
        completed=completed,
        on_trace=None,
        on_discard=None,
        warned_resources=warned,
    )
    seat.trainable = False
    harness = NullHarnessConfig(id="null", mcp_header_env={"world": {"X": "VAR"}})
    timeout = TimeoutConfig(rollout=5)

    run = seat.with_config(harness=harness, timeout=timeout, max_turns=3)

    assert type(run) is _EpisodeAgent and run is not seat
    assert run.config.harness == harness and run.harness.config == harness
    assert run.timeout == timeout and run.limits.max_turns == 3
    assert run.ctx.model == "m" and run.runtime_config == seat.runtime_config
    assert (run._name, run.trainable, run._gate) == ("solver", False, gate)
    assert run._completed is completed and run._warned_resources is warned
    assert seat.config.harness.id == "bash" and seat.limits.max_turns is None
    assert seat.with_config(timeout={"rollout": 5}).timeout == timeout
    with pytest.raises(ValidationError):
        seat.with_config(max_turn=3)
