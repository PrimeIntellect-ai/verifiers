import asyncio

import verifiers.v1 as vf
from verifiers.v1.agent import TimeoutConfig, _EpisodeAgent
from verifiers.v1.harnesses.null import NullHarnessConfig


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
