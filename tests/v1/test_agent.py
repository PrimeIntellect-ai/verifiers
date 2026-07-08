import verifiers.v1 as vf
from verifiers.v1.harnesses.null.harness import NullHarness, NullHarnessConfig
from verifiers.v1.rollout import Rollout


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
        return vf.Trace(task=self.task)

    monkeypatch.setattr(Rollout, "run", fake_run)
    agent = vf.Agent(
        NullHarness(NullHarnessConfig()),
        vf.ModelContext(model="org/model", client=object()),
        name="judge",
        trainable=False,
    )
    parent = vf.Trace(task=vf.Task(idx=0, prompt="seed"))
    task = vf.Task(idx=1, prompt="judge", sources=("source-trace",))

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
    assert trace.parents == ["source-trace"]
    assert trace.info["agent"]["runtime"]["borrowed"] is False


async def test_agent_provision_owns_runtime_lifetime(monkeypatch):
    runtime = FakeRuntime()
    import verifiers.v1.agent as agent_module

    monkeypatch.setattr(agent_module, "make_runtime", lambda config: runtime)
    agent = vf.Agent(
        NullHarness(NullHarnessConfig()),
        vf.ModelContext(model="org/model", client=object()),
    )

    async with agent.provision(vf.Task(idx=0, prompt="seed")) as box:
        assert box is runtime
        assert runtime.started
        assert not runtime.stopped

    assert runtime.stopped
