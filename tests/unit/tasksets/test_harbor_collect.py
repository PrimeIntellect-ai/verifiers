"""Harbor collect hooks: a check that refuses the submission the same way twice is a
`SubmissionRefused`; a failure that does not repeat, a signal, a missing command or a
timeout is a retryable failure, not a verdict on the submission."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.errors import SubmissionRefused
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.tasksets.harbor import taskset as harbor
from verifiers.v1.tasksets.harbor.taskset import (
    CollectHook,
    HarborData,
    HarborTask,
)


class HookRuntime:
    """Answers each run of the hook with the next exit status."""

    def __init__(self, *exits: int):
        self.exits = list(exits)
        self.runs = 0

    async def run(self, argv, env):
        self.runs += 1
        code = self.exits.pop(0)
        return ProgramResult(exit_code=code, stdout="", stderr=f"exit {code}\n")


def make_task() -> HarborTask:
    return HarborTask(
        HarborData(
            idx=0,
            prompt="p",
            task_dir="/nonexistent",
            collect=[CollectHook(command="check-submission")],
        )
    )


def make_trace(task: HarborTask) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="HarborTask", data=task.data),
    )


@pytest.fixture(autouse=True)
def no_artifacts(monkeypatch):
    async def collect(*args, **kwargs):
        return {}

    monkeypatch.setattr(harbor, "collect", collect)


async def test_a_check_that_fails_the_same_way_twice_refuses_the_submission():
    task = make_task()
    runtime = HookRuntime(2, 2)
    with pytest.raises(SubmissionRefused, match="exit 2, twice"):
        await task.finalize(make_trace(task), runtime)
    assert runtime.runs == 2


async def test_a_failure_that_passes_when_run_again_is_not_a_refusal():
    task = make_task()
    runtime = HookRuntime(1, 0)
    await task.finalize(make_trace(task), runtime)
    assert runtime.runs == 2


async def test_a_failure_that_changes_when_run_again_stays_retryable():
    task = make_task()
    runtime = HookRuntime(1, 2)
    with pytest.raises(RuntimeError, match="collect hook failed") as error:
        await task.finalize(make_trace(task), runtime)
    assert not isinstance(error.value, SubmissionRefused)


@pytest.mark.parametrize("code", [126, 127, 137, -1])
async def test_a_hook_that_could_not_run_or_was_killed_is_not_rerun(code):
    task = make_task()
    runtime = HookRuntime(code)
    with pytest.raises(RuntimeError, match="collect hook failed") as error:
        await task.finalize(make_trace(task), runtime)
    assert not isinstance(error.value, SubmissionRefused)
    assert runtime.runs == 1


def test_a_refusal_scores_the_verifier_reward_key_zero():
    task = make_task()
    trace = make_trace(task)
    task.record_refusal(trace)
    assert trace.rewards == {"reward": vf.Reward(score=0.0, weight=1.0)}
