"""Required scopes gate scoring only until preparation has completed."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import verifiers.v1.rollout as lifecycle
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.errors import TaskError
from verifiers.v1.rollout import Rollout, RolloutTimeouts
from verifiers.v1.runtimes import PrimeConfig, PrimeRuntimeInfo, RuntimeBackend
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.runtimes.process_scope import ProcessScope
from verifiers.v1.session import RolloutLimits
from verifiers.v1.task import Task
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask
from verifiers.v1.utils.decorators import reward


class ReadinessTask(Task[HarborData]):
    async def setup(self):
        await self.gate("task")

    async def finalize(self):
        self.events.append("finalize")

    @reward
    async def solved(self):
        self.events.append("score")
        return 1.0


class InertBackend(RuntimeBackend):
    SUPPORTS_TASK_IMAGES = True
    SUPPORTS_USER_SWITCHING = True

    def __init__(self):
        super().__init__("scope-readiness-test")
        self.config = PrimeConfig()
        self.info = PrimeRuntimeInfo()

    async def start(self):
        return

    async def exec(self, runtime, argv, env):
        raise AssertionError("no real process may run")

    async def run(self, runtime, argv, env):
        return ProgramResult(0, "", "")

    async def write(self, runtime, path, data):
        raise AssertionError("no real files may be staged")


@pytest.fixture
def readiness_rollout(monkeypatch):
    def make(*, required=True, blocked=None):
        entered = asyncio.Event()
        events = []

        async def gate(phase):
            if blocked == phase:
                entered.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    events.append("cancelled")
                    raise

        async def prepare(runtime):
            await gate("scope")
            return ProcessScope(runtime)

        async def resolve(runtime, user):
            return runtime.with_user(user, "/home/worker")

        async def setup(runtime):
            await gate("harness")

        async def noop(*args, **kwargs):
            return

        async def session(*args, **kwargs):
            return SimpleNamespace(close=noop)

        @asynccontextmanager
        async def interception(*args, **kwargs):
            yield "http://localhost:8000", "synthetic-model", "synthetic-state"

        @asynccontextmanager
        async def tools(*args, **kwargs):
            yield {}

        backend = InertBackend()
        monkeypatch.setattr(lifecycle, "make_backend", lambda *a, **kw: backend)
        monkeypatch.setattr(lifecycle, "resolve_agent_runtime", resolve)
        monkeypatch.setattr(ProcessScope, "prepare", prepare)
        monkeypatch.setattr(lifecycle, "serve_interception", interception)
        monkeypatch.setattr(lifecycle, "serve_tools", tools)
        task = ReadinessTask(
            HarborData(
                idx=0,
                prompt="synthetic",
                agent_user="worker",
                verifiers_process_scope=required,
            )
        )
        task.gate, task.events = gate, events
        harness = SimpleNamespace(
            config=SimpleNamespace(name="rlm"),
            SUPPORTS_TOOL_INTERCEPTION=False,
            replay_view=lambda message, dialect: message,
            setup=setup,
            session=session,
            score=noop,
            cleanup=noop,
        )
        run = Rollout(
            task=task,
            agent_config=AgentConfig(),
            harness=harness,
            ctx=SimpleNamespace(),
            runtime_config=backend.config,
            timeouts=RolloutTimeouts(setup=0.05 if blocked else None),
            limits=RolloutLimits(),
        )
        return run, backend, entered, events

    return make


@pytest.mark.parametrize("blocked", ["task", "scope"])
async def test_setup_timeout_before_scope_readiness_skips_scoring(
    readiness_rollout, blocked
):
    run, backend, entered, events = readiness_rollout(blocked=blocked)
    assert not await run.open()
    assert entered.is_set() and "cancelled" in events
    assert backend.cache.process_scope is None
    trace = await run.close()
    assert trace.stop_condition == "setup_timeout" and trace.is_timeout
    assert not trace.errors and trace.ok and not trace.rewards
    assert not trace.nodes
    assert trace.info["scoring_skipped"] == "workload_scope_unprepared"
    assert events.count("finalize") == 1 and "score" not in events
    assert backend.stopped


@pytest.mark.parametrize("blocked", ["task", "scope"])
async def test_external_cancellation_before_scope_readiness_aborts(
    readiness_rollout, blocked
):
    run, backend, entered, events = readiness_rollout(blocked=blocked)
    opening = asyncio.create_task(run.open())
    await entered.wait()
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert run.closed and backend.stopped
    assert backend.cache.process_scope is None
    assert "cancelled" in events and "score" not in events
    assert not run.trace.rewards


@pytest.mark.parametrize(
    "required,blocked,stop",
    [
        (True, None, None),
        (True, None, "agent"),
        (True, "harness", "setup"),
        (False, "task", "setup"),
    ],
)
async def test_prepared_scope_and_ordinary_timeouts_still_grade(
    readiness_rollout, required, blocked, stop
):
    run, backend, _entered, events = readiness_rollout(
        required=required, blocked=blocked
    )
    assert await run.open() is (blocked is None)
    if blocked is None and stop is not None:
        run.timeout(stop)
    trace = await run.close()
    assert trace.stop_condition == (f"{stop}_timeout" if stop else None)
    assert not trace.errors and trace.ok
    assert "scoring_skipped" not in trace.info
    assert events.count("score") == 1 and trace.reward == 1.0
    assert (backend.cache.process_scope is not None) is required
    assert backend.stopped


async def test_direct_harbor_verifier_keeps_fail_closed_scope_guard():
    task = HarborTask(HarborData(verifiers_process_scope=True))
    with pytest.raises(TaskError, match="prepared workload scope"):
        await task.run_verifier(InertBackend().bind(), None)
