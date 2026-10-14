"""NeMo Gym tasks: what their state needs on a run that continues a trace."""

import httpx

import verifiers.v1 as vf
from verifiers.v1.state import state_cls
from verifiers.v1.tasksets.nemo_gym.taskset import (
    NeMoGymData,
    NeMoGymTask,
    NeMoGymTaskConfig,
)
from verifiers.v1.tasksets.nemo_gym.toolset import NeMoGymState
from verifiers.v1.trace import AgentInfo, Trace, TraceTask


async def test_a_restored_run_rebuilds_its_gym_session(monkeypatch):
    """`trace.state` is never recorded: a continuation's `restore` seeds a Gym session
    and fills the state the tools and the verifier read, as `setup` does."""
    posted: list[str] = []

    async def post(self, path, body):
        posted.append(path)
        request = httpx.Request("POST", f"{self.resources_url}/{path}")
        return httpx.Response(
            200, json={}, headers={"set-cookie": "session=s1"}, request=request
        )

    monkeypatch.setattr(NeMoGymState, "post", post)
    tool = {"type": "function", "name": "lookup", "parameters": {}}
    row = {"responses_create_params": {"input": "hi", "tools": [tool]}}
    task = NeMoGymTask(
        NeMoGymData(prompt="hi", row=row),
        NeMoGymTaskConfig(resources_url="http://gym/"),
    )
    trace = Trace(
        task=TraceTask.of(task),
        state=state_cls(type(task))(),
        agent=AgentInfo(config=vf.AgentConfig(model="m")),
    )
    await task.restore(trace, None)
    assert posted == ["seed_session"]
    assert trace.state.resources_url == "http://gym"
    assert trace.state.tool_names == ["lookup"]
    assert trace.state.cookies == {"session": "s1"}
