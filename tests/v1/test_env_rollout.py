"""The env-rollout surface: roles()/rollout()/score() defaults, episode minting, role
stamping, crash-safe recording, and the score() deadline (no live agents — stubs stand
in behind `_agents_for`)."""

import asyncio

import pytest

import verifiers.v1 as vf
from verifiers.v1.trace import Trace, TraceTask


def _env_config(**kwargs) -> vf.SingleAgentEnvConfig:
    return vf.SingleAgentEnvConfig(taskset={"id": "echo-v1"}, **kwargs)


def _task(env: vf.Environment) -> vf.Task:
    return env.taskset.load()[0]


class StubAgent:
    """Duck-types the one method `_RoleAgent` wraps: mint a trace, publish it via
    `on_trace` (as the engine does the moment a run starts), return it finished."""

    def __init__(self, error: Exception | None = None) -> None:
        self.runs = 0
        self.error = error
        self.trainable = True  # the env-owned standing `brief()` adjusts

    async def run(
        self, task: vf.Task, *, runtime=None, shared_tools=None, on_trace=None
    ) -> Trace:
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
    env.brief(agents)  # the real _agents_for briefs after building
    env._agents_for = lambda ctx: agents  # type: ignore[method-assign]
    return agents


class DuetConfig(vf.EnvConfig):
    a: vf.AgentConfig = vf.AgentConfig()
    b: vf.AgentConfig = vf.AgentConfig()


def _duet_config(**kwargs) -> "DuetConfig":
    return DuetConfig(taskset={"id": "echo-v1"}, **kwargs)


class DuetEnv(vf.Environment[DuetConfig]):
    def brief(self, agents):
        agents["b"].trainable = False

    async def rollout(self, task, agents):
        a, b = await asyncio.gather(agents["a"].run(task), agents["b"].run(task))
        return {"a": a, "b": b}

    async def score(self, task, views):
        for trace in views.values():
            trace.record_metric("siblings", float(len(views)))


async def test_single_agent_env_mints_single_agent_records():
    """`SingleAgentEnv` is the single-agent case every plain taskset resolves to:
    one `agent` seat playing the seed taskset, one unstamped trace per episode,
    score() a no-op."""
    env = vf.SingleAgentEnv(_env_config())
    assert list(env._roles) == ["agent"]
    agents = _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert episode.ok and episode.env == "echo-v1"
    assert agents["agent"].runs == 1
    assert len(episode.traces) == 1
    trace = episode.traces[0]
    assert trace.role is None and trace.trainable  # the wire matches a plain eval's
    assert episode.task.data.idx == trace.task.data.idx


def test_roles_are_the_declared_agent_fields():
    """Roles are the AgentConfig fields on the env's config — the field name is the
    role, the only naming site. Every env stamps its traces' roles except
    `SingleAgentEnv`, whose wire stays identical to a plain eval's."""
    env = DuetEnv(_duet_config())
    assert list(env._roles) == ["a", "b"]  # declaration order
    assert all(isinstance(c, vf.AgentConfig) for c in env._roles.values())
    assert env._stamp_roles
    assert not vf.SingleAgentEnv(_env_config())._stamp_roles


def test_role_fields_must_declare_default_instances():
    """A `Field(default_factory=...)` role would silently fall out of role discovery
    and the CLI deep-merge; the config class refuses at definition instead."""
    from pydantic import Field

    with pytest.raises(TypeError, match="default instance"):

        class BadParams(vf.EnvConfig):
            solver: vf.AgentConfig = Field(default_factory=vf.AgentConfig)


def test_role_fields_must_not_shadow_base_fields():
    """A role named like a base `EnvConfig` field (`taskset`, `timeout`, ...) would
    break every framework read of that name; refuse at class definition."""
    with pytest.raises(TypeError, match="shadow"):

        class Shadowed(vf.EnvConfig):
            taskset: vf.AgentConfig = vf.AgentConfig()  # type: ignore[assignment]


async def test_multi_role_records_stamp_roles():
    env = DuetEnv(_duet_config())
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
    env = vf.SingleAgentEnv(_env_config())
    env._agents_for = lambda ctx: {"agent": StubAgent(error=RuntimeError("boom"))}  # type: ignore[method-assign]
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok and not episode.errors  # the failure lives on the trace
    assert episode.traces[0].error is not None


async def test_hook_crash_keeps_completed_traces():
    """A rollout() that raises after some runs finished still yields a episode carrying
    them — the crash-safe subset — with the failure on the episode, not a trace."""

    class Crashy(vf.SingleAgentEnv):
        async def rollout(self, task, agents):
            await agents["agent"].run(task)
            raise RuntimeError("hook bug")

    env = Crashy(_env_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok
    # Hook failures land episode-level with the stable boundary type.
    assert episode.error is not None and episode.error.type == "EnvError"
    assert "hook bug" in episode.error.message
    assert len(episode.traces) == 1 and episode.traces[0].error is None


async def test_score_deadline_is_a_record_error():
    class Slow(vf.SingleAgentEnv):
        async def score(self, task, views):
            await asyncio.sleep(60)

    env = Slow(_env_config(timeout={"score": 0.05}))
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok
    assert episode.error is not None and episode.error.type == "TimeoutError"
    assert len(episode.traces) == 1  # the finished traces survive the score failure


async def test_score_failure_keeps_the_views():
    """Once rollout() returns its views, they decide episode membership — a score()
    failure must not demote the episode to the completed buffer. The record
    flattens in mapping order (physical, not semantic)."""

    class Reordered(DuetEnv):
        async def rollout(self, task, agents):
            a = await agents["a"].run(task)
            b = await agents["b"].run(task)
            return {"b": b, "a": a}

        async def score(self, task, views):
            raise RuntimeError("judge crashed")

    env = Reordered(_duet_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok
    assert episode.error is not None and episode.error.type == "EnvError"
    assert [t.role for t in episode.traces] == ["b", "a"]


async def test_score_failure_keeps_rollout_membership():
    """rollout() may deliberately return a subset (a dropped warm-up, a forfeited
    seat); a score() failure keeps that membership, not the completed buffer."""

    class Subset(DuetEnv):
        async def rollout(self, task, agents):
            await agents["a"].run(task)  # ran, but the hook drops it
            return {"b": await agents["b"].run(task)}

        async def score(self, task, views):
            raise RuntimeError("judge crashed")

    env = Subset(_duet_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok
    assert [t.role for t in episode.traces] == ["b"]


async def test_views_must_be_a_named_bag():
    """rollout() returns a mapping of named views; any other shape is the
    env-rollout failing, recorded on the episode."""

    class Listy(vf.SingleAgentEnv):
        async def rollout(self, task, agents):
            return [await agents["agent"].run(task)]  # the retired list shape

    env = Listy(_env_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok and episode.error is not None
    assert "local views as a mapping" in episode.error.message


async def test_a_view_may_fan_out():
    """A list-valued view is a fanned-out seat: all its traces land on the episode."""

    class Fan(DuetEnv):
        async def rollout(self, task, agents):
            return {
                "a": [await agents["a"].run(task) for _ in range(2)],
                "b": await agents["b"].run(task),
            }

        async def score(self, task, views):
            pass

    env = Fan(_duet_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert episode.ok
    assert [t.role for t in episode.traces] == ["a", "a", "b"]


async def test_decorated_signals_cross_agent():
    """`@vf.reward`/`@vf.metric` on an Environment run in the default score(): once
    per target trace (`role=` narrows to one role's traces, unset is every trace),
    with the finished sibling set in reach; metrics record before rewards run, and
    reward weights apply."""

    class Signals(vf.Environment[DuetConfig]):
        async def rollout(self, task, agents):
            a, b = await asyncio.gather(agents["a"].run(task), agents["b"].run(task))
            return {"a": a, "b": b}

        @vf.metric(role="a")
        async def b_count(self, traces):
            return float(sum(t.role == "b" for t in traces))

        @vf.reward(weight=0.5)
        async def team(self, trace, traces):
            return 1.0

        @vf.reward(role="a")
        async def sees_metrics(self, trace):
            return trace.metrics["b_count"]  # metrics recorded before rewards run

    env = Signals(_duet_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert episode.ok
    a, b = episode.traces
    assert a.metrics == {"b_count": 1.0}
    assert a.rewards == {"team": 0.5, "sees_metrics": 1.0}
    assert b.metrics == {} and b.rewards == {"team": 0.5}


def test_decorated_signal_role_must_be_declared():
    class Bad(vf.SingleAgentEnv):
        @vf.metric(role="ghost")
        async def lost(self, traces):
            return 0.0

    with pytest.raises(ValueError, match="ghost"):
        Bad(_env_config())


async def test_decorated_signals_on_unstamped_single_role():
    """A `SingleAgentEnv` subclass leaves traces unstamped (the wire matches a plain
    eval's); a role='agent' signal still records onto them — every trace belongs to
    the sole implicit role."""

    class Solo(vf.SingleAgentEnv):
        @vf.metric(role="agent")
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

    class Bad(vf.SingleAgentEnv):
        @vf.metric
        async def broken(self, trace):
            raise RuntimeError("signal bug")

    env = Bad(_env_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok
    assert episode.error is not None and episode.error.type == "EnvError"
    assert len(episode.traces) == 1 and episode.traces[0].error is None


def test_roles_must_be_nonempty():
    class Empty(vf.Environment):
        async def rollout(self, task, agents):
            return {}

    with pytest.raises(ValueError, match="declares no roles"):
        Empty(vf.EnvConfig(taskset={"id": "echo-v1"}))


def test_slots_need_a_rollout():
    env = vf.SingleAgentEnv(_env_config())
    with pytest.raises(ValueError, match="n >= 1"):
        env.slots(_task(env), n=0)


async def test_run_slot_observes_and_completes():
    """`slots` plans n independent env-rollouts; `run_slot` runs each to its episode,
    keeping the slot live (traces appear at mint) and firing `on_complete` once final."""
    from verifiers.v1.trace import Episode

    env = vf.SingleAgentEnv(_env_config())
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
    env = vf.SingleAgentEnv(_env_config())
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
    env = vf.SingleAgentEnv(_env_config(max_turns=7, max_output_tokens=100))
    limits = env._role_limits(vf.AgentConfig(max_turns=2))
    assert limits.max_turns == 2  # the role's own cap wins
    assert limits.max_output_tokens == 100  # unset caps stay the env's


def test_role_harness_config_narrows_by_id():
    spec = vf.AgentConfig(harness={"id": "bash"})
    assert type(spec.harness) is not vf.HarnessConfig  # resolved to the concrete type
    assert spec.harness.id == "bash"


def test_seat_harness_is_a_pin_or_the_taskset_default():
    """There is no run-level harness: a seat runs its own pin
    (`--env.<role>.harness.*`), and an unpinned seat resolves to the taskset's
    default harness — never an operator-set run value."""
    assert vf.AgentConfig().harness is None
    env = vf.SingleAgentEnv(_env_config(agent={"harness": {"id": "null"}}))
    assert env._harnesses["agent"].config.id == "null"
    duet = DuetEnv(_duet_config())
    assert duet._harnesses["a"].config.id == "bash"  # the taskset's default
    assert duet._harnesses["a"] is duet._harnesses["b"]  # one object per config


def test_retired_top_level_axes_point_home():
    """The old flat axes are refused with a pointer to their new home — a config
    can't silently half-apply."""
    from verifiers.v1.configs.eval import EvalConfig

    with pytest.raises(ValueError, match="--env.taskset.id"):
        EvalConfig.model_validate({"taskset": {"id": "echo-v1"}})
    with pytest.raises(ValueError, match="--env.agent.harness"):
        EvalConfig.model_validate(
            {"env": {"taskset": {"id": "echo-v1"}}, "harness": {"id": "null"}}
        )


def test_role_harness_override_switches_the_type():
    """`--env.<role>.harness.id X` swaps the harness even over a pinned default: a
    discriminator switch replaces the subtree, so the old type's fields don't leak
    into the new type's (extra-forbidden) validation."""

    class Pinned(vf.EnvConfig):
        solver: vf.AgentConfig = vf.AgentConfig(harness=vf.HarnessConfig(id="bash"))

    params = Pinned.model_validate({"solver": {"harness": {"id": "null"}}})
    assert params.solver.harness is not None and params.solver.harness.id == "null"
    # A non-switching partial override still tunes the pinned harness in place.
    tuned = Pinned.model_validate({"solver": {"harness": {"edit": False}}})
    assert tuned.solver.harness is not None and tuned.solver.harness.id == "bash"
    assert tuned.solver.harness.edit is False


def test_role_pins_survive_partial_overrides():
    """A partial role override (`--env.user.sampling.temperature 0.7`) must not
    silently reset the role's declared pins — the field-default instance deep-merges
    under the provided keys, and only an explicit override replaces a pin."""

    class Pinned(vf.EnvConfig):
        user: vf.AgentConfig = vf.AgentConfig(model="frozen", max_turns=3)

    params = Pinned.model_validate({"user": {"sampling": {"temperature": 0.7}}})
    assert params.user.model == "frozen" and params.user.max_turns == 3
    assert params.user.sampling is not None
    assert params.user.sampling.temperature == 0.7
    explicit = Pinned.model_validate({"user": {"model": "other"}})
    assert explicit.user.model == "other" and explicit.user.max_turns == 3


def test_env_subclass_loads_and_config_narrows():
    """A taskset shipping an `Environment` subclass (duet-v1): the env config
    narrows to the declared config class from plain data (the `@ file.toml` /
    worker-rebuild / run-config paths alike) and `load_environment` constructs the
    subclass; plain tasksets resolve to `SingleAgentEnv`."""
    from verifiers.v1.configs.eval import EvalConfig

    config_cls = vf.env_config_type("duet-v1")
    cfg = vf.resolve_env_config(
        {"taskset": {"id": "duet-v1"}, "b": {"model": "frozen"}}
    )
    assert isinstance(cfg, config_cls)
    assert cfg.b.model == "frozen"
    assert cfg.b.harness is not None
    assert cfg.b.harness.id == "null"  # the pin survives the partial override
    env = vf.load_environment(cfg)
    assert type(env).__name__ == "DuetEnv" and list(env._roles) == ["a", "b"]
    # a run config narrows its `env` field the same way
    run = EvalConfig(env={"taskset": {"id": "duet-v1"}})
    assert isinstance(run.env, config_cls)
    assert vf.environment_class("echo-v1") is vf.SingleAgentEnv
    assert type(vf.load_environment(_env_config())) is vf.SingleAgentEnv


def test_env_requires_its_declared_config():
    """Constructing an env whose config wasn't narrowed to its config class fails
    loudly instead of breaking later in role discovery."""
    with pytest.raises(TypeError, match="declares Environment\\[DuetConfig\\]"):
        DuetEnv(_env_config())


def test_env_config_data_keeps_the_env_shape():
    """The env-server wire dict keeps the env's roles and knobs, and the worker-side
    rebuild re-narrows them to the concrete config class."""
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.serve.pool import env_config_data

    config = EvalConfig(env={"taskset": {"id": "duet-v1"}, "a": {"max_turns": 3}})
    data = env_config_data(config.env)
    rebuilt = vf.resolve_env_config(data)
    assert isinstance(rebuilt, vf.env_config_type("duet-v1"))
    assert rebuilt.a.max_turns == 3
    assert rebuilt.a.harness.id == "null"  # the fixture's pin rode the wire


async def test_max_concurrent_gates_env_internal_fanout():
    """The semaphore bounds agent RUNS, not episodes — an env's internal fan-out
    (asyncio.gather inside rollout()) counts against --max-concurrent too."""
    env = DuetEnv(_duet_config())
    state = {"live": 0, "peak": 0}

    class Gauged(StubAgent):
        async def run(self, task, *, runtime=None, shared_tools=None, on_trace=None):
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
            await asyncio.sleep(0.02)
            state["live"] -= 1
            return await super().run(
                task, runtime=runtime, shared_tools=shared_tools, on_trace=on_trace
            )

    agents = {name: Gauged() for name in env._roles}
    env._agents_for = lambda ctx: agents  # type: ignore[method-assign]
    episode = await env.run_episode(_task(env), None, gate=asyncio.Semaphore(1))
    assert episode.ok and state["peak"] == 1


def test_role_scoped_signals_belong_to_environments():
    """`role=` routes an Environment's cross-trace signals; on a Task or Harness it
    would be silently unscoped — refused at class definition instead."""
    with pytest.raises(TypeError, match="role="):

        class BadTask(vf.Task):
            @vf.reward(role="solver")
            async def scoped(self, trace):
                return 0.0

    with pytest.raises(TypeError, match="role="):

        class BadHarness(vf.Harness):
            @vf.metric(role="solver")
            async def scoped(self, trace):
                return 0.0


async def test_aliased_views_land_once():
    """A trace named under two view keys (a natural authoring move — e.g. `winner`
    aliasing a solver) is one run: serialized once, scored once."""

    class Aliased(vf.Environment[DuetConfig]):
        async def rollout(self, task, agents):
            a = await agents["a"].run(task)
            return {"a": a, "winner": a, "b": await agents["b"].run(task)}

        @vf.metric
        async def n(self, trace):
            return 1.0

    env = Aliased(_duet_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert episode.ok and len(episode.traces) == 2
    assert all(t.metrics.get("n") == 1.0 for t in episode.traces)


async def test_empty_views_fail_the_rollout():
    """A rollout() that returns no traces is the env-rollout failing, not an ok
    empty episode that resume would keep forever."""

    class Empty(vf.SingleAgentEnv):
        async def rollout(self, task, agents):
            return {}

    env = Empty(_env_config())
    _stub_agents(env)
    episode = await env.run_episode(_task(env), None)
    assert not episode.ok and episode.error is not None
    assert "returned no traces" in episode.error.message


def test_scoring_handlers_must_be_async():
    """A sync handler would surface as an opaque asyncio error at score time,
    attributed to the scoring stage; refused at definition instead."""
    for deco in (vf.reward, vf.metric, vf.stop):
        with pytest.raises(TypeError, match="async def"):

            @deco
            def sync_handler(self, trace):
                return 0.0
