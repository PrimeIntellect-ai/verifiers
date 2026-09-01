from contextlib import asynccontextmanager
from types import SimpleNamespace

import verifiers.v1 as vf
from verifiers.v1.envs.isolated_verifier import (
    IsolatedVerifierEnv,
    IsolatedVerifierEnvConfig,
)
from verifiers.v1.envs.isolated_verifier import env as isolated_verifier
from verifiers.v1.tasksets.harbor.env import HarborEnv, HarborEnvConfig
from verifiers.v1.tasksets.harbor.taskset import (
    HarborData,
    HarborTask,
    VerifierConfig,
)


class VerifierTask(vf.Task):
    controller_state: list[str]

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        self.controller_state.append(runtime.name)
        runtime.events.append(f"setup:{runtime.name}")

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        runtime.events.append(f"finalize:{runtime.name}")

    @vf.reward
    async def verified(self, runtime: vf.Runtime) -> float:
        runtime.events.append(f"score:{runtime.name}")
        return float(runtime.name == "grader" and self.controller_state == ["grader"])


class CustomizedHarborTask(HarborTask):
    @vf.metric
    async def extra_metric(self) -> float:
        return 1.0

    async def run_verifier(
        self, runtime: vf.Runtime, trace: vf.Trace
    ) -> float | dict[str, float]:
        return {"harbor": 1.0}


class FakeRuntime:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.stopped = False

    async def prepare_setup(self) -> None:
        self.events.append(f"prepare-setup:{self.name}")

    async def prepare_execution(self, routes: list[str]) -> None:
        assert routes == []
        self.events.append(f"prepare-execution:{self.name}")


class FakeAgent:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.runtime = FakeRuntime("solver", events)
        self.solution: vf.Trace | None = None

    @asynccontextmanager
    async def provision(self, task: vf.Task):
        self.events.append("start:solver")
        try:
            yield self.runtime
        finally:
            self.runtime.stopped = True
            self.events.append("stop:solver")

    async def run(self, task: vf.Task, *, runtime: FakeRuntime) -> vf.Trace:
        trace = vf.Trace(
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            task=vf.TraceTask(type=type(task).__name__, data=task.data),
        )
        await task.setup(trace, runtime)
        await task.finalize(trace, runtime)
        await task.score(trace, runtime)
        trace.ok = trace.is_completed = True
        self.solution = trace
        return trace


async def test_isolated_verifier_lifecycle(monkeypatch) -> None:
    events: list[str] = []
    task = VerifierTask(
        vf.TaskData(
            idx=0,
            prompt="solve",
            artifacts=[vf.Artifact(source="answer.txt")],
        )
    )
    task.controller_state = []
    agent = FakeAgent(events)
    env = object.__new__(IsolatedVerifierEnv)
    env.config = IsolatedVerifierEnvConfig(
        agent=vf.AgentConfig(runtime=vf.DockerConfig()),
        verifier_env={"PHASE": "verifier"},
    )

    async def collect(runtime, artifacts):
        assert runtime is agent.runtime and not runtime.stopped
        assert artifacts == task.data.artifacts
        events.append("collect:solver")
        return {"/app/answer.txt": b"answer"}

    async def restore(runtime, artifacts):
        assert runtime.name == "grader"
        assert artifacts == {"/app/answer.txt": b"answer"}
        events.append("restore:grader")

    @asynccontextmanager
    async def provision_runtime(config, name=None, env=None):
        assert agent.runtime.stopped
        assert isinstance(config, vf.DockerConfig)
        assert env == {"PHASE": "verifier"}
        grader = FakeRuntime("grader", events)
        events.append("start:grader")
        try:
            yield grader
        finally:
            grader.stopped = True
            events.append("stop:grader")

    monkeypatch.setattr(isolated_verifier.vf, "collect", collect)
    monkeypatch.setattr(isolated_verifier.vf, "restore", restore)
    monkeypatch.setattr(isolated_verifier, "provision_runtime", provision_runtime)

    await env.run(task, SimpleNamespace(agent=agent))
    assert agent.solution is not None
    episode = vf.Episode(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        traces=[agent.solution],
    )
    await env.finalize(task, episode)

    assert events == [
        "start:solver",
        "setup:solver",
        "finalize:solver",
        "collect:solver",
        "stop:solver",
        "start:grader",
        "prepare-setup:grader",
        "setup:grader",
        "restore:grader",
        "prepare-execution:grader",
        "score:grader",
        "stop:grader",
    ]
    assert agent.solution.rewards["verified"].score == 1.0
    assert agent.solution.state.artifacts == {"/app/answer.txt": b"answer"}
    assert task.controller_state == []


def test_isolated_verifier_accepts_independent_runtime() -> None:
    task = vf.Task(
        vf.TaskData(
            idx=0,
            prompt="solve",
            image="solver:latest",
            network_allow=["solver.example"],
        )
    )
    verifier_runtime = vf.DockerConfig(
        image="verifier:latest",
        workdir="/verify",
        cpu=2,
        memory=4,
        allow=["verifier.example"],
    )
    env = object.__new__(IsolatedVerifierEnv)
    env.config = IsolatedVerifierEnvConfig(
        agent=vf.AgentConfig(runtime=vf.DockerConfig()),
        verifier_runtime=verifier_runtime,
    )

    assert env.verifier_config(task) == verifier_runtime


async def test_harbor_keeps_extra_scoring_with_solver(monkeypatch) -> None:
    task = CustomizedHarborTask(
        HarborData(
            idx=0,
            prompt="solve",
            verifier=VerifierConfig(),
        )
    )
    agent = FakeAgent([])
    env = object.__new__(HarborEnv)
    env.config = HarborEnvConfig(agent=vf.AgentConfig(runtime=vf.DockerConfig()))
    monkeypatch.setattr(
        HarborEnv, "verifier_config", lambda self, task: vf.DockerConfig()
    )

    async def run(task, runtime=None):
        trace = vf.Trace(
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            task=vf.TraceTask(type=type(task).__name__, data=task.data),
        )
        await task.score(trace, FakeRuntime("solver", []))
        trace.ok = trace.is_completed = True
        agent.solution = trace
        return trace

    monkeypatch.setattr(agent, "run", run)
    await env.run(task, SimpleNamespace(agent=agent))

    assert agent.solution is not None
    assert agent.solution.metrics == {"extra_metric": 1.0}
    assert agent.solution.rewards == {}


async def test_harbor_fresh_verifier_runs_only_builtin_score() -> None:
    task = CustomizedHarborTask(
        HarborData(idx=0, prompt="solve", verifier=VerifierConfig())
    )
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
    )
    env = object.__new__(HarborEnv)

    scores = await env.verify(task, trace, FakeRuntime("grader", []))

    assert scores == {"harbor": 1.0}
    assert trace.rewards == {}
    assert trace.metrics == {}
