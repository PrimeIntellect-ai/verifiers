import pytest

import verifiers.v1 as vf
from verifiers.v1.env import validate_task_pairing
from verifiers.v1.errors import TaskError
from verifiers.v1.harnesses.null.harness import NullHarness, NullHarnessConfig
from verifiers.v1.rollout import Rollout
from verifiers.v1.trace import TraceTask


def task_of(prompt) -> vf.Task:
    return vf.Task(vf.TaskData(idx=0, prompt=prompt))


def trace_for(task: vf.Task) -> vf.Trace:
    return vf.Trace(task=TraceTask(type=type(task).__name__, data=task.data))


def null_agent(**kwargs) -> vf.Agent:
    return vf.Agent(
        NullHarness(NullHarnessConfig()),
        vf.ModelContext(
            model="org/model", client=object(), sampling=vf.SamplingConfig()
        ),
        **kwargs,
    )


class FakeRuntime:
    def __init__(self, config=None) -> None:
        self.config = config or vf.SubprocessConfig()
        self.name = "fake-runtime"
        self.descriptor = "fake"
        self.info = None
        self.stopped = False
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


async def test_agent_run_stamps_lineage_and_borrowed_runtime(monkeypatch):
    async def fake_run(self):
        self.runtime = self._borrowed_runtime
        return trace_for(self.task)

    monkeypatch.setattr(Rollout, "run", fake_run)
    agent = vf.Agent(
        NullHarness(NullHarnessConfig()),
        vf.ModelContext(
            model="org/model", client=object(), sampling=vf.SamplingConfig()
        ),
        name="judge",
        trainable=False,
    )
    parent = trace_for(task_of("seed"))
    task = vf.Task(vf.TaskData(idx=1, prompt="judge"))

    trace = await agent.run(task, parents=[parent], runtime=FakeRuntime())
    assert trace.agent == "judge"
    assert trace.parents == [parent.id]
    assert trace.trainable is False
    assert trace.info["agent"]["model"] == "org/model"
    assert trace.info["agent"]["runtime"] == {
        "type": "subprocess",
        "descriptor": "fake",
        "borrowed": True,
    }

    trace = await agent.run(task)
    assert trace.parents == []
    assert trace.info["agent"]["runtime"]["borrowed"] is False


async def test_agent_provision_owns_runtime_lifetime(monkeypatch):
    runtime = FakeRuntime()
    import verifiers.v1.agent as agent_module

    monkeypatch.setattr(agent_module, "make_runtime", lambda config: runtime)
    agent = vf.Agent(
        NullHarness(NullHarnessConfig()),
        vf.ModelContext(
            model="org/model", client=object(), sampling=vf.SamplingConfig()
        ),
    )

    async with agent.provision(task_of("seed")) as box:
        assert box is runtime
        assert runtime.started
        assert not runtime.stopped

    assert runtime.stopped


async def test_borrowing_a_torn_down_runtime_raises():
    """Borrow-after-teardown is a lifetime bug in the borrowing program: refused up
    front, at the caller, instead of failing opaquely mid-harness."""
    box = FakeRuntime()
    await box.stop()
    with pytest.raises(ValueError, match="already torn down"):
        await null_agent().run(task_of("hi"), runtime=box)


async def test_mid_run_teardown_of_borrowed_box_raises_to_caller():
    """The owner tearing the box down under an in-flight run is the same lifetime bug,
    surfaced through the same channel: the failure is re-attributed and raised to the
    caller (raw error chained), never captured as a world error onto the trace."""

    class SabotagedTask(vf.Task):
        async def setup(self, trace: vf.Trace, runtime) -> None:
            await runtime.stop()  # the owner tearing the box down under the run
            raise RuntimeError("box died under the harness")

    with pytest.raises(ValueError, match="mid-run") as excinfo:
        await null_agent().run(
            SabotagedTask(vf.TaskData(idx=0, prompt="hi")), runtime=FakeRuntime()
        )
    assert isinstance(excinfo.value.__cause__, TaskError)  # raw failure chained


async def test_needs_container_checks_the_resolved_runtime(monkeypatch):
    """NEEDS_CONTAINER is validated against where the run actually lands: a borrowed
    docker box satisfies it even under a subprocess-defaulting harness, and resolving
    to the harness's subprocess policy still refuses."""

    class BoxedTask(vf.Task):
        NEEDS_CONTAINER = True

    validate_task_pairing(
        NullHarness(NullHarnessConfig()), BoxedTask, vf.DockerConfig()
    )  # the resolved runtime is what counts, not the harness's subprocess default

    async def fake_run(self):
        self.runtime = self._borrowed_runtime
        return trace_for(self.task)

    monkeypatch.setattr(Rollout, "run", fake_run)
    agent = null_agent()
    task = BoxedTask(vf.TaskData(idx=0, prompt="hi"))
    await agent.run(task, runtime=FakeRuntime(vf.DockerConfig()))  # borrowed box: fine
    with pytest.raises(ValueError, match="NEEDS_CONTAINER"):
        await agent.run(task)  # resolves to the harness's subprocess policy
