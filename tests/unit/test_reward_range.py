"""Reward ranges: the scoring policy's checks, skip rewards, the trace's pass verdict,
and the episode-level range check."""

import pytest

import verifiers.v1 as vf
import verifiers.v1.agent as agent_module
from verifiers.v1.configs.scoring import resolve_scoring, weighted_reward_range


def make_trace(**kwargs) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(
            type="Task", data=vf.TaskData(idx=0, prompt=None), key="k", hash="h"
        ),
        **kwargs,
    )


def test_threshold_defaults_to_the_range_high():
    assert vf.ScoringConfig().threshold == 1.0
    assert vf.ScoringConfig(reward_range=(-1.0, 3.0)).threshold == 3.0
    assert vf.ScoringConfig(pass_threshold=0.8).threshold == 0.8
    assert vf.ScoringConfig(reward_range=(0.0, None)).threshold is None


def test_range_low_must_be_below_its_high():
    with pytest.raises(ValueError, match="must be below"):
        vf.ScoringConfig(reward_range=(1.0, 1.0))
    vf.ScoringConfig(reward_range=(None, None))


def test_skip_reward_falls_back_to_the_range_low():
    policy = vf.ScoringConfig(
        reward_range=(-1.0, 1.0), skip_rewards={"truncated": -0.5}
    )
    assert policy.skip_reward("truncated") == -0.5
    assert policy.skip_reward("format_violation") == -1.0
    unbounded = vf.ScoringConfig(reward_range=(None, 1.0))
    with pytest.raises(ValueError, match="skip_rewards\\['timeout'\\]"):
        unbounded.skip_reward("timeout")


@pytest.mark.parametrize(
    "policy, match",
    [
        (vf.ScoringConfig(pass_threshold=0.0), "must lie in"),
        (vf.ScoringConfig(pass_threshold=1.5), "must lie in"),
        (vf.ScoringConfig(skip_rewards={"timeout": -0.1}), "outside reward_range"),
        (vf.ScoringConfig(skip_rewards={"timeout": 1.0}), "would count as a pass"),
        (
            vf.ScoringConfig(reward_range=(None, 1.0), skip_on_truncated=True),
            "skip_rewards for \\['truncated'\\]",
        ),
    ],
)
def test_resolved_policy_checks(policy, match):
    with pytest.raises(ValueError, match=match):
        resolve_scoring(policy, vf.ScoringConfig())


def test_agent_overrides_merge_before_the_check():
    task = vf.ScoringConfig(reward_range=(0.0, 10.0), skip_rewards={"timeout": 1.0})
    # Alone, the agent's threshold exceeds its own default range; merged, it fits.
    agent = vf.ScoringConfig(pass_threshold=5.0, skip_rewards={"truncated": 2.0})
    resolved = resolve_scoring(task, agent)
    assert resolved.threshold == 5.0
    assert resolved.skip_rewards == {"timeout": 1.0, "truncated": 2.0}


def test_weighted_reward_range():
    assert weighted_reward_range((1.0, (0.0, 1.0)), (0.5, (0.0, 1.0))) == (0.0, 1.5)
    assert weighted_reward_range((0.0, (None, None)), (1.0, (0.0, 1.0))) == (0.0, 1.0)
    assert weighted_reward_range((-2.0, (0.0, 1.0))) == (-2.0, 0.0)
    assert weighted_reward_range((1.0, (None, 1.0)), (1.0, (0.0, 1.0))) == (None, 2.0)
    assert weighted_reward_range((-1.0, (None, 1.0))) == (-1.0, None)


def test_trace_passed_and_in_range():
    trace = make_trace(reward_range=(-1.0, 1.0), pass_threshold=0.5)
    trace.record_reward("score", 0.5)
    assert trace.passed and trace.in_reward_range
    trace.record_reward("bonus", 1.0)
    assert not trace.in_reward_range
    assert make_trace(pass_threshold=None).passed is None


async def test_rescore_keeps_a_skip_reward():
    task = vf.Task(vf.TaskData(idx=0, prompt=None))
    trace = make_trace(ok=True)
    trace.record_reward("skipped/truncated", -1.0)
    trace.record_metric("stale", 1.0)
    await task.rescore(trace)
    assert set(trace.rewards) == {"skipped/truncated"}
    assert trace.reward == -1.0
    assert trace.metrics == {}


class OneAgentConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig(model="test-model")


class OneAgentEnv(vf.Env[OneAgentConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        await agents.agent.run(task)


@pytest.mark.parametrize("reward, ok", [(1.0, True), (2.0, False)])
async def test_episode_fails_on_an_out_of_range_reward(monkeypatch, reward, ok):
    class FakeRollout:
        def __init__(self, *, on_trace, trace_id, attempt, **_):
            self.trace = make_trace(id=trace_id, attempt=attempt, ok=True)
            self.trace.record_reward("score", reward)
            on_trace(self.trace)

        async def open(self) -> bool:
            return False

        async def close(self) -> vf.Trace:
            return self.trace

    monkeypatch.setattr(agent_module, "Rollout", FakeRollout)
    monkeypatch.setattr(agent_module.Agent, "_rollout_params", lambda *args: {})
    env = OneAgentEnv(OneAgentConfig(taskset={"id": "reverse-text"}))
    ctx = vf.ModelContext(model="test-model", client=vf.EvalClientConfig())
    episode = await env.run_episode(vf.Task(vf.TaskData(idx=0, prompt=None)), ctx)
    assert episode.ok is ok
    if not ok:
        (error,) = episode.errors
        assert error.type == "EnvError"
        assert "outside reward_range [0.0, 1.0]" in error.message
