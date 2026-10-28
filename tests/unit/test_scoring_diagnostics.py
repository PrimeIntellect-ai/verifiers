"""Bounded diagnostics for shared and isolated task scoring."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.errors import SandboxError, TaskError
from verifiers.v1.grading import grade_task, scoring_phase, scoring_transaction
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask


def make_trace(task: vf.Task) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
    )


class Backend:
    async def prepare_setup(self):
        pass

    async def prepare_execution(self, routes):
        pass


class VerifierRuntime:
    def __init__(self):
        self.backend = Backend()
        self.env = {}

    def with_env(self, env):
        self.env = env
        return self


class BlockingBackend(Backend):
    def __init__(self):
        self.started = asyncio.Event()

    async def prepare_setup(self):
        self.started.set()
        await asyncio.Event().wait()


class Deployment:
    def __init__(self, runtime):
        self.runtime = runtime

    async def prepare_verifier(self):
        pass

    @asynccontextmanager
    async def verifier(self, config, env):
        yield self.runtime


class DiagnosticTask(vf.Task):
    @vf.reward
    async def solved(self, runtime) -> float:
        return 1.0


class RewardRuntime:
    def __init__(self, *, reward: str = "1", error: Exception | None = None):
        self.reward = reward
        self.error = error

    async def run(self, command, env):
        if command == ["bash", "/tests/test.sh"]:
            return SimpleNamespace(exit_code=0, stdout="", stderr="")
        # reward.json is absent, which permits the legacy text fallback.
        return SimpleNamespace(exit_code=1, stdout="", stderr="")

    async def read(self, path, max_bytes):
        if self.error is not None:
            raise self.error
        return self.reward.encode()


async def test_isolated_scoring_persists_context_and_setup_phases():
    task = DiagnosticTask(vf.TaskData(prompt=""))
    trace = make_trace(task)
    await grade_task(
        task,
        trace,
        None,
        vf.TaskTimeout(setup=5, scoring=20),
        vf.ScoringConfig(mode="isolated", retries=RetryConfig(max_retries=0)),
        Deployment(VerifierRuntime()),
        scoring_elapsed=1.25,
    )

    assert trace.rewards["solved"].score == 1.0
    diagnostics = trace.info["scoring_diagnostics"]
    assert diagnostics["budget"] == 20
    assert diagnostics["mode"] == "isolated"
    assert diagnostics["metrics_elapsed"] == 1.25
    assert [(event["phase"], event["status"]) for event in diagnostics["events"]] == [
        ("isolated_setup", "completed"),
        ("staging", "completed"),
    ]


async def test_isolated_setup_cancellation_survives_the_copy():
    task = DiagnosticTask(vf.TaskData(prompt=""))
    trace = make_trace(task)
    runtime = VerifierRuntime()
    runtime.backend = BlockingBackend()
    pending = asyncio.create_task(
        grade_task(
            task,
            trace,
            None,
            vf.TaskTimeout(setup=5, scoring=20),
            vf.ScoringConfig(mode="isolated", retries=RetryConfig(max_retries=0)),
            Deployment(runtime),
        )
    )
    await runtime.backend.started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    [event] = trace.info["scoring_diagnostics"]["events"]
    assert event["phase"] == "isolated_setup"
    assert event["status"] == "cancelled"
    assert event["error_type"] == "CancelledError"


async def test_harbor_text_reward_records_execution_and_reads():
    task = HarborTask(HarborData(prompt=""))
    trace = make_trace(task)

    assert await task.run_verifier(RewardRuntime(), trace) == {"reward": 1.0}
    events = trace.info["scoring_diagnostics"]["events"]
    assert [event["phase"] for event in events] == [
        "verifier_execution",
        "reward_read",
        "reward_read",
    ]
    assert all(event["status"] == "completed" for event in events)
    assert all("stdout" not in event and "stderr" not in event for event in events)


async def test_harbor_missing_reward_reports_bounded_fallback_error():
    task = HarborTask(HarborData(prompt=""))
    trace = make_trace(task)
    runtime = RewardRuntime(error=SandboxError("read failed: " + "x" * 1000))

    with pytest.raises(TaskError) as raised:
        await task.run_verifier(runtime, trace)

    message = str(raised.value)
    fallback = message.split("reward.txt fallback: ", 1)[1]
    assert fallback.startswith("SandboxError: read failed: ")
    assert len(fallback) <= 500
    events = trace.info["scoring_diagnostics"]["events"]
    assert events[-1]["phase"] == "reward_read"
    assert events[-1]["status"] == "failed"
    assert events[-1]["error_type"] == "SandboxError"


async def test_cancellation_survives_the_scoring_transaction():
    trace = make_trace(DiagnosticTask(vf.TaskData(prompt="")))
    started = asyncio.Event()

    async def score():
        with scoring_transaction(trace):
            async with scoring_phase(trace, "reward_read"):
                started.set()
                await asyncio.Event().wait()

    pending = asyncio.create_task(score())
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    [event] = trace.info["scoring_diagnostics"]["events"]
    assert event["phase"] == "reward_read"
    assert event["status"] == "cancelled"
    assert event["error_type"] == "CancelledError"


async def test_scoring_phase_events_are_bounded():
    trace = make_trace(DiagnosticTask(vf.TaskData(prompt="")))
    for _ in range(40):
        async with scoring_phase(trace, "reward_read"):
            pass

    diagnostics = trace.info["scoring_diagnostics"]
    assert len(diagnostics["events"]) == 32
    assert diagnostics["events_dropped"] == 8
