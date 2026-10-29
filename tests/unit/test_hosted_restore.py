"""A continuation on the live box its run left (`Box.hosted`) takes the task as that run
left it, unless the run failed on the box's side (`BOX_SIDE_FAILURES`): whatever killed
its harness may have taken the task's services with it, so the continuation restores
the task (`Task.restore`)."""

import pytest
from test_agent_resume import (  # noqa: F401 - fixtures
    EVENTS,
    NotesTask,
    ScriptHarness,
    fake_boxes,
    make,
    model,
)

import verifiers.v1 as vf
from verifiers.v1.runtimes import ProgramResult


class KilledOnce(ScriptHarness):
    """Killed by a signal on its first launch (a grader killing every process of its
    user, the box's memory limit), a normal tool loop after."""

    launches = 0

    async def launch(self, *args, **kwargs) -> ProgramResult:
        KilledOnce.launches += 1
        if KilledOnce.launches == 1:
            return ProgramResult(exit_code=137, stdout="", stderr="Killed")
        return await super().launch(*args, **kwargs)


def task() -> NotesTask:
    return NotesTask(vf.TaskData(idx=0, prompt="start"))


@pytest.fixture(autouse=True)
def boxes(fake_boxes):  # noqa: F811
    KilledOnce.launches = 0


async def test_a_continuation_after_a_killed_harness_restores_the_task(model):  # noqa: F811
    agent = make(model, harness=KilledOnce)
    async with agent.provision(task()) as box:
        first = await agent.run(task(), runtime=box)
        assert first.last_error is not None and first.last_error.type == "HarnessKilled"
        second = await agent.run(task(), resume=first, note="go on", runtime=box)
    assert second.ok, second.errors
    assert EVENTS == ["setup", "restore"]


async def test_a_continuation_after_a_clean_run_takes_the_box_as_it_was(model):  # noqa: F811
    agent = make(model)
    async with agent.provision(task()) as box:
        first = await agent.run(task(), runtime=box)
        assert first.ok, first.errors
        second = await agent.run(task(), resume=first, note="go on", runtime=box)
    assert second.ok, second.errors
    assert EVENTS == ["setup"]
