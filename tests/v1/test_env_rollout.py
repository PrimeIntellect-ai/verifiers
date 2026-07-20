"""The env-rollout surface: role discovery, rollout()/score() defaults, episode
minting, role stamping, crash-safe recording, and the score() deadline (no live
runs — the engine is stubbed at `Agent.run`, under the real per-episode agents)."""

import asyncio
from collections import Counter

import pytest

import verifiers.v1 as vf
from verifiers.v1.agent import Agent
from verifiers.v1.trace import AgentInfo, Trace, TraceTask


def _env_config(**kwargs) -> vf.SingleAgentEnvConfig:
    return vf.SingleAgentEnvConfig(taskset={"id": "echo-v1"}, **kwargs)


def _task(env: vf.Environment) -> vf.Task:
    return env.taskset.load()[0]


def _ctx() -> vf.ModelContext:
    """A run context for stubbed-engine tests (duck client — no call ever leaves)."""
    return vf.ModelContext(model="stub", client=object())  # type: ignore[arg-type]


def _stub_engine(
    env: vf.Environment,
    monkeypatch,
    errors: dict[str, Exception] | None = None,
) -> Counter:
    """Stub the engine underneath the real per-episode agents: `Agent.run` mints a
    finished trace through the injected watch, so agent construction, `brief()`,
    role stamping, gating, and crash-safe capture all run for real. Returns the
    per-role run counter; `errors` injects a failure onto a role's traces."""
    errs = dict(errors or {})
    runs: Counter = Counter()

    async def fake_run(self, task, *, runtime=None, shared_tools=None, on_trace=None):
        runs[self._name] += 1
        # Mirrors the real engine: RolloutRun mints the trace with its AgentInfo
        # already attached, then fires on_trace (where the episode stamps land).
        trace = Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            agent=AgentInfo(model=self.ctx.model),
        )
        if on_trace is not None:
            on_trace(trace)
        error = errs.get(self._name)
        if error is not None:
            trace.capture_error(error)
        trace.is_completed = True
        return trace

    monkeypatch.setattr(Agent, "run", fake_run)
    return runs


class DuetConfig(vf.EnvConfig):
    a: vf.AgentConfig = vf.AgentConfig()
    b: vf.AgentConfig = vf.AgentConfig()


def _duet_config(**kwargs) -> "DuetConfig":
    return DuetConfig(taskset={"id": "echo-v1"}, **kwargs)


class DuetEnv(vf.Environment[DuetConfig]):
    def brief(self, agents):
        agents["b"].trainable = False

    async def rollout(self, task, agents):
        await asyncio.gather(agents["a"].run(task), agents["b"].run(task))

    async def score(self, task, traces):
        for trace in traces:
            trace.record_metric("siblings", float(len(traces)))


async def test_single_agent_env_mints_single_agent_records(monkeypatch):
    """`SingleAgentEnv` is the single-agent case every plain taskset resolves to:
    one `agent` seat playing the seed taskset, one unstamped trace per episode,
    score() a no-op."""
    env = vf.SingleAgentEnv(_env_config())
    assert list(env._roles) == ["agent"]
    runs = _stub_engine(env, monkeypatch)
    episode = await env.run_episode(_task(env), _ctx())
    assert episode.ok and episode.env == "echo-v1"
    assert runs["agent"] == 1
    assert len(episode.traces) == 1
    trace = episode.traces[0]
    assert (
        trace.agent_name is None and trace.trainable
    )  # the wire matches a plain eval's
    assert episode.task.data.idx == trace.task.data.idx


async def test_multi_role_records_stamp_roles(monkeypatch):
    env = DuetEnv(_duet_config())
    _stub_engine(env, monkeypatch)
    seen_live: list[str | None] = []
    episode = await env.run_episode(
        _task(env), _ctx(), on_trace=lambda t: seen_live.append(t.agent_name)
    )
    assert episode.ok and len(episode.traces) == 2
    assert [t.agent_name for t in episode.traces] == ["a", "b"]
    assert [t.trainable for t in episode.traces] == [True, False]
    # score() saw the finished sibling set; stamps were already live at mint
    assert all(t.metrics["siblings"] == 2.0 for t in episode.traces)
    assert sorted(seen_live) == ["a", "b"]


async def test_agent_failures_are_trace_data_not_record_errors(monkeypatch):
    env = vf.SingleAgentEnv(_env_config())
    _stub_engine(env, monkeypatch, errors={"agent": RuntimeError("boom")})
    episode = await env.run_episode(_task(env), _ctx())
    assert not episode.ok and not episode.errors  # the failure lives on the trace
    assert episode.traces[0].error is not None


async def test_hook_crash_keeps_completed_traces(monkeypatch):
    """A rollout() that raises after some runs finished still yields a episode carrying
    them — the crash-safe subset — with the failure on the episode, not a trace."""

    class Crashy(vf.SingleAgentEnv):
        async def rollout(self, task, agents):
            await agents["agent"].run(task)
            raise RuntimeError("hook bug")

    env = Crashy(_env_config())
    _stub_engine(env, monkeypatch)
    episode = await env.run_episode(_task(env), _ctx())
    assert not episode.ok
    # Hook failures land episode-level with the stable boundary type.
    assert episode.error is not None and episode.error.type == "EnvError"
    assert "hook bug" in episode.error.message
    assert len(episode.traces) == 1 and episode.traces[0].error is None


async def test_score_deadline_is_a_record_error(monkeypatch):
    class Slow(vf.SingleAgentEnv):
        async def score(self, task, views):
            await asyncio.sleep(60)

    env = Slow(_env_config(timeout={"score": 0.05}))
    _stub_engine(env, monkeypatch)
    episode = await env.run_episode(_task(env), _ctx())
    assert not episode.ok
    assert episode.error is not None and episode.error.type == "TimeoutError"
    assert len(episode.traces) == 1  # the finished traces survive the score failure


async def test_score_failure_keeps_the_traces(monkeypatch):
    """Every completed run is the episode's data (completion order) — a score()
    failure lands on the episode's errors without demoting its traces."""

    class Judged(DuetEnv):
        async def score(self, task, traces):
            raise RuntimeError("judge crashed")

    env = Judged(_duet_config())
    _stub_engine(env, monkeypatch)
    episode = await env.run_episode(_task(env), _ctx())
    assert not episode.ok
    assert episode.error is not None and episode.error.type == "EnvError"
    assert [t.agent_name for t in episode.traces] == ["a", "b"]


async def test_empty_rollout_is_the_env_failing(monkeypatch):
    """A rollout() that runs no agent yields no episode data — refused as the
    env-rollout failing, on the episode's errors."""

    class Lazy(DuetEnv):
        async def rollout(self, task, agents):
            pass

    env = Lazy(_duet_config())
    _stub_engine(env, monkeypatch)
    episode = await env.run_episode(_task(env), _ctx())
    assert not episode.ok and episode.error is not None
    assert "ran no agent" in episode.error.message and episode.traces == []


async def test_a_seat_may_fan_out(monkeypatch):
    """Running a seat n times is a fanned-out seat: every completed run lands on
    the episode, all stamped with the same role."""

    class Fan(DuetEnv):
        async def rollout(self, task, agents):
            for _ in range(2):
                await agents["a"].run(task)
            await agents["b"].run(task)

        async def score(self, task, traces):
            pass

    env = Fan(_duet_config())
    _stub_engine(env, monkeypatch)
    episode = await env.run_episode(_task(env), _ctx())
    assert episode.ok
    assert [t.agent_name for t in episode.traces] == ["a", "a", "b"]


async def test_decorated_signals_cross_agent(monkeypatch):
    """`@vf.reward`/`@vf.metric` on an Environment run in the default score(): once
    per target trace (`role=` narrows to one role's traces, unset is every trace),
    with the finished sibling set in reach; metrics record before rewards run, and
    reward weights apply."""

    class Signals(vf.Environment[DuetConfig]):
        async def rollout(self, task, agents):
            await asyncio.gather(agents["a"].run(task), agents["b"].run(task))

        @vf.metric(agent="a")
        async def b_count(self, traces):
            return float(sum(t.agent_name == "b" for t in traces))

        @vf.reward(weight=0.5)
        async def team(self, trace, traces):
            return 1.0

        @vf.reward(agent="a")
        async def sees_metrics(self, trace):
            return trace.metrics["b_count"]  # metrics recorded before rewards run

    env = Signals(_duet_config())
    _stub_engine(env, monkeypatch)
    episode = await env.run_episode(_task(env), _ctx())
    assert episode.ok
    a, b = episode.traces
    assert a.metrics == {"b_count": 1.0}
    assert a.rewards == {"team": 0.5, "sees_metrics": 1.0}
    assert b.metrics == {} and b.rewards == {"team": 0.5}


def test_decorated_signal_role_must_be_declared():
    class Bad(vf.SingleAgentEnv):
        @vf.metric(agent="ghost")
        async def lost(self, traces):
            return 0.0

    with pytest.raises(ValueError, match="ghost"):
        Bad(_env_config())


async def test_decorated_signals_on_unstamped_single_role(monkeypatch):
    """A `SingleAgentEnv` subclass leaves traces unstamped (the wire matches a plain
    eval's); a role='agent' signal still records onto them — every trace belongs to
    the sole implicit role."""

    class Solo(vf.SingleAgentEnv):
        @vf.metric(agent="agent")
        async def n(self, traces):
            return float(len(traces))

    env = Solo(_env_config())
    _stub_engine(env, monkeypatch)
    episode = await env.run_episode(_task(env), _ctx())
    assert episode.traces[0].agent_name is None
    assert episode.traces[0].metrics["n"] == 1.0


async def test_decorated_signal_failure_is_an_episode_error(monkeypatch):
    """A decorated signal raising is score() failing — a rollout-level error on the
    episode, with the finished traces kept (never blamed on a trace)."""

    class Bad(vf.SingleAgentEnv):
        @vf.metric
        async def broken(self, trace):
            raise RuntimeError("signal bug")

    env = Bad(_env_config())
    _stub_engine(env, monkeypatch)
    episode = await env.run_episode(_task(env), _ctx())
    assert not episode.ok
    assert episode.error is not None and episode.error.type == "EnvError"
    assert len(episode.traces) == 1 and episode.traces[0].error is None


def test_slots_need_a_rollout():
    env = vf.SingleAgentEnv(_env_config())
    with pytest.raises(ValueError, match="n >= 1"):
        env.slots(_task(env), n=0)


async def test_run_slot_observes_and_completes(monkeypatch):
    """`slots` plans n independent env-rollouts; `run_slot` runs each to its episode,
    keeping the slot live (traces appear at mint) and firing `on_complete` once final."""
    from verifiers.v1.trace import Episode

    env = vf.SingleAgentEnv(_env_config())
    _stub_engine(env, monkeypatch)
    slots = env.slots(_task(env), n=3)
    assert [s.traces for s in slots] == [[]] * 3
    assert not any(s.done for s in slots)
    completed: list[Episode] = []

    async def on_complete(episode: Episode) -> None:
        completed.append(episode)

    episodes = [await env.run_slot(slot, _ctx(), None, on_complete) for slot in slots]
    assert [s.episode for s in slots] == episodes
    assert [s.traces for s in slots] == [list(r.traces) for r in episodes]
    assert all(s.done for s in slots)
    assert completed == episodes
    # --resume preloads kept episodes as already-finished slots.
    from verifiers.v1.env import RunSlot

    slot = RunSlot.finished(episodes[0])
    assert slot.done and slot.episode is episodes[0]
    assert slot.traces == list(episodes[0].traces)


def test_role_pins_fall_back_per_field():
    """What a rollout's agent actually gets: the role's pins where set (model,
    sampling) fall back to the run's per field, and the per-run caps are the
    seat's own — asserted on the built per-episode agents, not the helpers."""

    class PinnedConfig(vf.EnvConfig):
        a: vf.AgentConfig = vf.AgentConfig()
        b: vf.AgentConfig = vf.AgentConfig(
            model="frozen",
            max_turns=2,
            sampling=vf.SamplingConfig(temperature=0.0),
        )

    class PinnedEnv(vf.Environment[PinnedConfig]):
        async def rollout(self, task, agents):
            pass

    env = PinnedEnv(PinnedConfig(taskset={"id": "echo-v1"}, a={"max_turns": 7}))
    ctx = vf.ModelContext(model="run-model", client=object())  # duck client
    agents = env._episode_agents(ctx, "ep", None, [], None)
    assert agents["a"].ctx.model == "run-model"  # nothing pinned → the run's
    assert agents["a"].ctx.client is ctx.client
    assert agents["a"].limits.max_turns == 7  # the seat's own cap
    assert agents["a"].limits.max_output_tokens is None  # unset = no limit
    assert agents["b"].ctx.model == "frozen"  # pins win, per field
    assert agents["b"].ctx.client is ctx.client  # unpinned legs stay the run's
    assert agents["b"].ctx.sampling.temperature == 0.0
    assert agents["b"].limits.max_turns == 2


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


async def test_max_concurrent_gates_env_internal_fanout(monkeypatch):
    """The semaphore bounds agent RUNS, not episodes — an env's internal fan-out
    (asyncio.gather inside rollout()) counts against --max-concurrent too. The
    gate is acquired by the per-episode agent, so the gauge under the stubbed
    engine sees at most one run live."""
    env = DuetEnv(_duet_config())
    state = {"live": 0, "peak": 0}

    async def gauged_run(self, task, *, runtime=None, shared_tools=None, on_trace=None):
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])
        await asyncio.sleep(0.02)
        state["live"] -= 1
        trace = Trace(task=TraceTask(type=type(task).__name__, data=task.data))
        if on_trace is not None:
            on_trace(trace)
        trace.is_completed = True
        return trace

    monkeypatch.setattr(Agent, "run", gauged_run)
    episode = await env.run_episode(_task(env), _ctx(), gate=asyncio.Semaphore(1))
    assert episode.ok and state["peak"] == 1


def test_role_scoped_signals_belong_to_environments():
    """`role=` routes an Environment's cross-trace signals; on a Task or Harness it
    would be silently unscoped — refused at class definition instead."""
    with pytest.raises(TypeError, match="agent="):

        class BadTask(vf.Task):
            @vf.reward(agent="solver")
            async def scoped(self, trace):
                return 0.0

    with pytest.raises(TypeError, match="agent="):

        class BadHarness(vf.Harness):
            @vf.metric(agent="solver")
            async def scoped(self, trace):
                return 0.0


async def test_aliased_views_land_once(monkeypatch):
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
    _stub_engine(env, monkeypatch)
    episode = await env.run_episode(_task(env), _ctx())
    assert episode.ok and len(episode.traces) == 2
    assert all(t.metrics.get("n") == 1.0 for t in episode.traces)


def test_scoring_handlers_must_be_async():
    """A sync handler would surface as an opaque asyncio error at score time,
    attributed to the scoring stage; refused at definition instead."""
    for deco in (vf.reward, vf.metric, vf.stop):
        with pytest.raises(TypeError, match="async def"):

            @deco
            def sync_handler(self, trace):
                return 0.0
