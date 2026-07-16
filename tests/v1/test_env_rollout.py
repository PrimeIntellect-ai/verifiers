"""The env-rollout surface: roles()/rollout()/score() defaults, episode minting, role
stamping, crash-safe recording, and the score() deadline (no live agents — stubs stand
in behind `_agents_for`)."""

import asyncio

import pytest

import verifiers.v1 as vf
from verifiers.v1.trace import Trace, TraceTask


def _env_config(**kwargs) -> vf.EnvConfig:
    return vf.EnvConfig(taskset={"id": "echo-v1"}, **kwargs)


def _task(env: vf.Environment) -> vf.Task:
    return env.taskset.load()[0]


class StubAgent:
    """Duck-types the one method `_RoleAgent` wraps: mint a trace, publish it via
    `on_trace` (as the engine does the moment a run starts), return it finished."""

    def __init__(self, error: Exception | None = None) -> None:
        self.runs = 0
        self.error = error

    async def run(self, task: vf.Task, *, runtime=None, on_trace=None) -> Trace:
        self.runs += 1
        trace = Trace(task=TraceTask(type=type(task).__name__, data=task.data))
        if on_trace is not None:
            on_trace(trace)
        if self.error is not None:
            trace.capture_error(self.error)
        trace.is_completed = True
        return trace


def _stub_agents(env: vf.Environment) -> dict[str, StubAgent]:
    agents = {name: StubAgent() for name in env._roles}
    env._agents_for = lambda ctx: agents  # type: ignore[method-assign]
    return agents


class DuetParams(vf.EnvParams):
    a: vf.AgentConfig = vf.AgentConfig()
    b: vf.AgentConfig = vf.AgentConfig(trainable=False)


class DuetEnv(vf.Environment[DuetParams]):
    def roles(self):
        return {"a": vf.Role(self.params.a), "b": vf.Role(self.params.b)}

    async def rollout(self, task, agents):
        return list(await asyncio.gather(agents["a"].run(task), agents["b"].run(task)))

    async def score(self, task, traces):
        for trace in traces:
            trace.record_metric("siblings", float(len(traces)))


async def test_base_env_mints_single_agent_records():
    """The base defaults ARE the single-agent case: one 'main' role on the env's
    harness, one unstamped trace per episode, score() a no-op."""
    env = vf.Environment(_env_config())
    assert list(env._roles) == ["solver"]
    agents = _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert episode.ok and episode.env == "echo-v1"
    assert agents["solver"].runs == 1
    assert len(episode.traces) == 1
    trace = episode.traces[0]
    assert trace.role is None and trace.trainable  # the wire matches a plain eval's
    assert episode.task.data.idx == trace.task.data.idx


async def test_multi_role_records_stamp_roles():
    env = DuetEnv(_env_config(env=DuetParams()))
    _stub_agents(env)
    seen_live: list[str | None] = []
    episode = await env.run_episode(
        _task(env), None, on_trace=lambda t: seen_live.append(t.role)
    )
    assert episode.ok and len(episode.traces) == 2
    assert [t.role for t in episode.traces] == ["a", "b"]
    assert [t.trainable for t in episode.traces] == [True, False]
    # score() saw the finished sibling set; stamps were already live at mint
    assert all(t.metrics["siblings"] == 2.0 for t in episode.traces)
    assert sorted(seen_live) == ["a", "b"]


async def test_agent_failures_are_trace_data_not_record_errors():
    env = vf.Environment(_env_config())
    env._agents_for = lambda ctx: {"solver": StubAgent(error=RuntimeError("boom"))}  # type: ignore[method-assign]
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok and not episode.errors  # the failure lives on the trace
    assert episode.traces[0].error is not None


async def test_hook_crash_keeps_completed_traces():
    """A rollout() that raises after some runs finished still yields a episode carrying
    them — the crash-safe subset — with the failure on the episode, not a trace."""

    class Crashy(vf.Environment):
        async def rollout(self, task, agents):
            await agents["solver"].run(task)
            raise RuntimeError("hook bug")

    env = Crashy(_env_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok
    assert episode.error is not None and episode.error.type == "RuntimeError"
    assert len(episode.traces) == 1 and episode.traces[0].error is None


async def test_score_deadline_is_a_record_error():
    class Slow(vf.Environment):
        async def score(self, task, traces):
            await asyncio.sleep(60)

    env = Slow(_env_config(timeout={"score": 0.05}))
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok
    assert episode.error is not None and episode.error.type == "TimeoutError"
    assert len(episode.traces) == 1  # the finished traces survive the score failure


async def test_decorated_signals_cross_agent():
    """`@vf.reward`/`@vf.metric` on an Environment run in the default score(): once
    per target trace (`role=` narrows to one role's traces, unset is every trace),
    with the finished sibling set in reach; metrics record before rewards run, and
    reward weights apply."""

    class Signals(vf.Environment[DuetParams]):
        def roles(self):
            return {"a": vf.Role(self.params.a), "b": vf.Role(self.params.b)}

        async def rollout(self, task, agents):
            return list(
                await asyncio.gather(agents["a"].run(task), agents["b"].run(task))
            )

        @vf.metric(role="a")
        async def b_count(self, traces):
            return float(sum(t.role == "b" for t in traces))

        @vf.reward(weight=0.5)
        async def team(self, trace, traces):
            return 1.0

        @vf.reward(role="a")
        async def sees_metrics(self, trace):
            return trace.metrics["b_count"]  # metrics recorded before rewards run

    env = Signals(_env_config(env=DuetParams()))
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert episode.ok
    a, b = episode.traces
    assert a.metrics == {"b_count": 1.0}
    assert a.rewards == {"team": 0.5, "sees_metrics": 1.0}
    assert b.metrics == {} and b.rewards == {"team": 0.5}


def test_decorated_signal_role_must_be_declared():
    class Bad(vf.Environment):
        @vf.metric(role="ghost")
        async def lost(self, traces):
            return 0.0

    with pytest.raises(ValueError, match="ghost"):
        Bad(_env_config())


async def test_decorated_signals_on_unstamped_single_role():
    """An env subclass keeping the default roles() leaves traces unstamped (the wire
    matches a plain eval's); a role='solver' signal still records onto them — every
    trace belongs to the sole implicit role."""

    class Solo(vf.Environment):
        @vf.metric(role="solver")
        async def n(self, traces):
            return float(len(traces))

    env = Solo(_env_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert episode.traces[0].role is None
    assert episode.traces[0].metrics["n"] == 1.0


async def test_decorated_signal_failure_is_an_episode_error():
    """A decorated signal raising is score() failing — a rollout-level error on the
    episode, with the finished traces kept (never blamed on a trace)."""

    class Bad(vf.Environment):
        @vf.metric
        async def broken(self, trace):
            raise RuntimeError("signal bug")

    env = Bad(_env_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok
    assert episode.error is not None and episode.error.type == "RuntimeError"
    assert len(episode.traces) == 1 and episode.traces[0].error is None


def test_roles_must_be_nonempty():
    class Empty(vf.Environment):
        def roles(self):
            return {}

    with pytest.raises(ValueError, match="at least one agent"):
        Empty(_env_config())


def test_slots_need_a_rollout():
    env = vf.Environment(_env_config())
    with pytest.raises(ValueError, match="n >= 1"):
        env.slots(_task(env), n=0)


async def test_run_slot_observes_and_completes():
    """`slots` plans n independent env-rollouts; `run_slot` runs each to its episode,
    keeping the slot live (traces appear at mint) and firing `on_complete` once final."""
    from verifiers.v1.trace import Episode

    env = vf.Environment(_env_config())
    _stub_agents(env)
    slots = env.slots(_task(env), n=3)
    assert [s.traces for s in slots] == [[]] * 3
    assert not any(s.done for s in slots)
    completed: list[Episode] = []

    async def on_complete(episode: Episode) -> None:
        completed.append(episode)

    episodes = [await env.run_slot(slot, None, None, on_complete) for slot in slots]
    assert [s.episode for s in slots] == episodes
    assert [s.traces for s in slots] == [list(r.traces) for r in episodes]
    assert all(s.done for s in slots)
    assert completed == episodes


def test_finished_slot_from_saved_record():
    from verifiers.v1.env import RunSlot
    from verifiers.v1.trace import Episode

    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=7, prompt="hi")))
    episode = Episode.of(trace, env="stub")
    slot = RunSlot.finished(episode)
    assert slot.done and slot.episode is episode and slot.task.data.idx == 7
    assert slot.traces == [trace]


def test_role_ctx_pins_fall_back_per_field():
    env = vf.Environment(_env_config())
    ctx = vf.ModelContext(model="run-model", client=object())  # duck client
    assert env._role_ctx(vf.AgentConfig(), ctx) is ctx  # nothing pinned → the run's
    pinned = env._role_ctx(vf.AgentConfig(model="frozen-judge"), ctx)
    assert pinned.model == "frozen-judge"
    assert pinned.client is ctx.client  # unpinned legs stay the run's
    sampled = env._role_ctx(
        vf.AgentConfig(sampling=vf.SamplingConfig(temperature=0.0)), ctx
    )
    assert sampled.model == "run-model"
    assert sampled.sampling.temperature == 0.0


def test_role_limits_fall_back_per_field():
    env = vf.Environment(_env_config(max_turns=7, max_output_tokens=100))
    limits = env._role_limits(vf.AgentConfig(max_turns=2))
    assert limits.max_turns == 2  # the role's own cap wins
    assert limits.max_output_tokens == 100  # unset caps stay the env's


def test_role_harness_config_narrows_by_id():
    spec = vf.AgentConfig(harness={"id": "default"})
    assert type(spec.harness) is not vf.HarnessConfig  # resolved to the concrete type
    assert spec.harness.id == "default"


def test_role_harness_late_binds_to_the_runs():
    """An unpinned role plays on the run's `--harness.*` — pairing an env must not
    silently swap the policy's harness (the axes stay orthogonal)."""
    assert vf.AgentConfig().harness is None
    env = vf.Environment(_env_config(harness={"id": "null"}))
    assert env._harnesses["solver"] is env.harness
    assert env.harness.config.id == "null"


def test_role_harness_override_switches_the_type():
    """`--env.<role>.harness.id X` swaps the harness even over a pinned default: a
    discriminator switch replaces the subtree, so the old type's fields don't leak
    into the new type's (extra-forbidden) validation."""

    class Pinned(vf.EnvParams):
        solver: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="default"))

    params = Pinned.model_validate({"solver": {"harness": {"id": "null"}}})
    assert params.solver.harness is not None and params.solver.harness.id == "null"
    # A non-switching partial override still tunes the pinned harness in place.
    tuned = Pinned.model_validate({"solver": {"harness": {"edit": False}}})
    assert tuned.solver.harness is not None and tuned.solver.harness.id == "default"
    assert tuned.solver.harness.edit is False


def test_role_pins_survive_partial_overrides():
    """A partial role override (`--env.user.sampling.temperature 0.7`) must not
    silently reset the role's declared pins — the field-default instance deep-merges
    under the provided keys, and only an explicit override replaces a pin."""

    class Pinned(vf.EnvParams):
        user: vf.AgentConfig = vf.AgentConfig(model="frozen", trainable=False)

    params = Pinned.model_validate({"user": {"sampling": {"temperature": 0.7}}})
    assert params.user.model == "frozen" and params.user.trainable is False
    assert params.user.sampling is not None
    assert params.user.sampling.temperature == 0.7
    explicit = Pinned.model_validate({"user": {"model": "other"}})
    assert explicit.user.model == "other" and explicit.user.trainable is False


def test_env_subclass_loads_and_params_narrow():
    """A taskset shipping an `Environment` subclass (duet-v1): the `env` field narrows
    to the declared params type from plain data (the `@ file.toml` / worker-rebuild /
    direct-constructor paths alike) and `load_environment` constructs the subclass;
    base tasksets stay base."""
    params_cls = vf.env_params_type("duet-v1")
    cfg = vf.EnvConfig.model_validate(
        {"taskset": {"id": "duet-v1"}, "env": {"b": {"model": "frozen"}}}
    )
    assert isinstance(cfg.env, params_cls)
    assert cfg.env.b.model == "frozen"
    assert cfg.env.b.trainable is False  # the pin survives the partial override
    env = vf.load_environment(cfg)
    assert type(env).__name__ == "DuetEnv" and list(env._roles) == ["a", "b"]
    # the direct-constructor path narrows too (no model_validate detour needed)
    constructed = vf.EnvConfig(taskset={"id": "duet-v1"})
    assert isinstance(constructed.env, params_cls)
    assert vf.environment_class("echo-v1") is vf.Environment
    assert type(vf.load_environment(_env_config())) is vf.Environment


def test_env_requires_its_declared_params():
    """Constructing an env whose config wasn't narrowed to its params type fails
    loudly instead of breaking later in `roles()`."""
    with pytest.raises(TypeError, match="declares Environment\\[DuetParams\\]"):
        DuetEnv(_env_config())


def test_env_config_data_keeps_env_params():
    """The env-server wire dict keeps the env params (dropping eval-only fields), and
    the worker-side rebuild re-narrows them."""
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.serve.pool import env_config_data

    config = EvalConfig(taskset={"id": "duet-v1"}, env={"a": {"max_turns": 3}})
    data = env_config_data(config)
    assert "env" in data and "num_rollouts" not in data
    rebuilt = vf.EnvConfig.model_validate(data)
    assert isinstance(rebuilt.env, vf.env_params_type("duet-v1"))
    assert rebuilt.env.a.max_turns == 3
    assert rebuilt.env.a.harness.id == "null"  # the fixture's pin rode the wire
