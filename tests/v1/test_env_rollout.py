"""The env-rollout surface: roles()/rollout()/score() defaults, record minting, role
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
        return {"a": self.params.a, "b": self.params.b}

    async def rollout(self, task, agents):
        return list(await asyncio.gather(agents["a"].run(task), agents["b"].run(task)))

    async def score(self, task, traces):
        for trace in traces:
            trace.record_metric("siblings", float(len(traces)))


async def test_base_env_mints_single_agent_records():
    """The base defaults ARE the single-agent case: one 'main' role on the env's
    harness, one unstamped trace per record, score() a no-op."""
    env = vf.Environment(_env_config())
    assert list(env._roles) == ["main"]
    agents = _stub_agents(env)
    record = await env.run_record(_task(env), None)
    assert record.ok and record.env == "echo-v1"
    assert agents["main"].runs == 1
    assert len(record.traces) == 1
    trace = record.traces[0]
    assert trace.role is None and trace.trainable  # the wire matches a plain eval's
    assert record.task.data.idx == trace.task.data.idx


async def test_multi_role_records_stamp_roles():
    env = DuetEnv(_env_config(env=DuetParams()))
    _stub_agents(env)
    seen_live: list[str | None] = []
    record = await env.run_record(
        _task(env), None, on_trace=lambda t: seen_live.append(t.role)
    )
    assert record.ok and len(record.traces) == 2
    assert [t.role for t in record.traces] == ["a", "b"]
    assert [t.trainable for t in record.traces] == [True, False]
    # score() saw the finished sibling set; stamps were already live at mint
    assert all(t.metrics["siblings"] == 2.0 for t in record.traces)
    assert sorted(seen_live) == ["a", "b"]


async def test_agent_failures_are_trace_data_not_record_errors():
    env = vf.Environment(_env_config())
    env._agents_for = lambda ctx: {"main": StubAgent(error=RuntimeError("boom"))}  # type: ignore[method-assign]
    record = await env.run_record(_task(env), None)
    assert not record.ok and not record.errors  # the failure lives on the trace
    assert record.traces[0].error is not None


async def test_hook_crash_keeps_completed_traces():
    """A rollout() that raises after some runs finished still yields a record carrying
    them — the crash-safe subset — with the failure on the record, not a trace."""

    class Crashy(vf.Environment):
        async def rollout(self, task, agents):
            await agents["main"].run(task)
            raise RuntimeError("hook bug")

    env = Crashy(_env_config())
    _stub_agents(env)
    record = await env.run_record(_task(env), None)
    assert not record.ok
    assert record.error is not None and record.error.type == "RuntimeError"
    assert len(record.traces) == 1 and record.traces[0].error is None


async def test_score_deadline_is_a_record_error():
    class Slow(vf.Environment):
        async def score(self, task, traces):
            await asyncio.sleep(60)

    env = Slow(_env_config(timeout={"score": 0.05}))
    _stub_agents(env)
    record = await env.run_record(_task(env), None)
    assert not record.ok
    assert record.error is not None and record.error.type == "TimeoutError"
    assert len(record.traces) == 1  # the finished traces survive the score failure


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
    """`slots` plans n independent env-rollouts; `run_slot` runs each to its record,
    keeping the slot live (traces appear at mint) and firing `on_complete` once final."""
    from verifiers.v1.trace import RolloutRecord

    env = vf.Environment(_env_config())
    _stub_agents(env)
    slots = env.slots(_task(env), n=3)
    assert [s.traces for s in slots] == [[]] * 3
    assert not any(s.done for s in slots)
    completed: list[RolloutRecord] = []

    async def on_complete(record: RolloutRecord) -> None:
        completed.append(record)

    records = [await env.run_slot(slot, None, None, on_complete) for slot in slots]
    assert [s.record for s in slots] == records
    assert [s.traces for s in slots] == [list(r.traces) for r in records]
    assert all(s.done for s in slots)
    assert completed == records


def test_finished_slot_from_saved_record():
    from verifiers.v1.env import RunSlot
    from verifiers.v1.trace import RolloutRecord

    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=7, prompt="hi")))
    record = RolloutRecord.of(trace, env="stub")
    slot = RunSlot.finished(record)
    assert slot.done and slot.record is record and slot.task.data.idx == 7
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
