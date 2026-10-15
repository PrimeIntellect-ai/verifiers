"""A task's boundary hooks may take the rollout's runtime: to inspect the box at a model
call, or checkpoint it there (`Runtime.checkpoint`, recorded on the trace)."""

import pytest
from test_agent_resume import (  # noqa: F401 - fixtures
    NotesTask,
    ScriptHarness,
    fake_boxes,
)
from test_provider_retry import Provider, make

import verifiers.v1 as vf


class Snapshots(NotesTask):
    """Checkpoints its box on each model response and lists what it held then."""

    @vf.intercept
    async def snapshot(
        self, response: vf.Response, trace: vf.Trace, runtime: vf.Runtime
    ) -> None:
        assert runtime.supports_checkpoint
        turns = trace.num_turns + (0 if response.message.tool_calls else 1)
        checkpoint = await runtime.checkpoint(trace, turns)
        listed = await runtime.run(["sh", "-c", "echo listed"], {})
        trace.info.setdefault("listed", {})[checkpoint.id] = listed.stdout.strip()


@pytest.mark.usefixtures("fake_boxes")
async def test_a_response_hook_checkpoints_the_box_and_runs_in_it():
    async with Provider() as provider:
        agent = make(provider, ScriptHarness)
        trace = await agent.run(Snapshots(vf.TaskData(idx=0, prompt="start")))
    assert trace.ok, trace.errors
    own = [c for c in trace.checkpoints if not c.restored]
    assert own and len(own) == len(provider.requests)
    assert set(trace.info["listed"]) == {c.id for c in own}
    assert all(value == "listed" for value in trace.info["listed"].values())
