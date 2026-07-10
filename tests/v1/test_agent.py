import verifiers.v1 as vf
from verifiers.v1.harnesses.null.harness import NullHarness, NullHarnessConfig
from verifiers.v1.rollout import Rollout
from verifiers.v1.trace import TraceTask


def task_of(prompt) -> vf.Task:
    return vf.Task(vf.TaskData(idx=0, prompt=prompt))


def trace_for(task: vf.Task) -> vf.Trace:
    return vf.Trace(task=TraceTask(type=type(task).__name__, data=task.data))


class FakeRuntime:
    def __init__(self) -> None:
        self.config = vf.SubprocessConfig()
        self.name = "fake-runtime"
        self.descriptor = "fake"
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
