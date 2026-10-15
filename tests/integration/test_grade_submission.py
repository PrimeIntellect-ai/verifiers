"""`grade_submission(on_score=...)`: a model-free grading's post-scoring callback runs
in the fresh verifier, and only an attempt it finishes commits its scores."""

import shutil

import pytest

import verifiers.v1 as vf
from verifiers.v1.errors import TaskError
from verifiers.v1.runtimes import provision_runtime

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not shutil.which("docker"), reason="needs Docker"),
]

IMAGE = "python:3.11-slim"


class AnswerTask(vf.Task):
    @vf.reward
    async def correct(self, runtime: vf.Runtime) -> float:
        return float((await runtime.read("/app/answer")).strip() == b"42")


def answer_task() -> AnswerTask:
    return AnswerTask(
        vf.TaskData(
            prompt="Write 42 to /app/answer",
            image=IMAGE,
            workdir="/app",
            scoring=vf.ScoringConfig(
                mode="isolated", retries=vf.RetryConfig(max_retries=0)
            ),
            artifacts=[vf.Artifact(source="/app/answer")],
        )
    )


async def solved_submission(task: AnswerTask, runtime: vf.Runtime) -> vf.Trace:
    """The solver's box after a solve: the answer, and a file outside the artifacts."""
    await runtime.run(["sh", "-c", "mkdir -p /app && echo 42 > /app/answer"], {})
    await runtime.write("/var/tmp/solver-only", b"solver")
    trace = vf.Trace(
        task=vf.TraceTask.of(task),
        agent=vf.AgentInfo(config=vf.AgentConfig(model="test/fake-model")),
    )
    trace.state.artifacts = await vf.collect(runtime, task.data.artifacts)
    return trace


async def test_on_score_runs_in_the_verifier_and_its_reward_commits():
    task = answer_task()
    seen: list[tuple[int, int]] = []

    async def judge(trace: vf.Trace, runtime: vf.Runtime) -> None:
        answer = await runtime.run(["test", "-f", "/app/answer"], {})
        solver_only = await runtime.run(["test", "-e", "/var/tmp/solver-only"], {})
        seen.append((answer.exit_code, solver_only.exit_code))
        assert trace.rewards["correct"].value == 1.0
        trace.record_reward("judge", 0.5)

    config = vf.DockerConfig(image=IMAGE, workdir="/app")
    async with provision_runtime(config) as runtime:
        trace = await solved_submission(task, runtime)
        await vf.grade_submission(task, trace, runtime, on_score=judge)

    # The callback saw the restored submission in a fresh box, not the solver's.
    assert seen == [(0, 1)]
    assert trace.rewards["correct"].value == 1.0
    assert trace.rewards["judge"].value == 0.5


async def test_a_failing_on_score_commits_no_score():
    task = answer_task()

    async def judge(trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.record_reward("judge", 1.0)
        raise RuntimeError("the judge gave no verdict")

    config = vf.DockerConfig(image=IMAGE, workdir="/app")
    async with provision_runtime(config) as runtime:
        trace = await solved_submission(task, runtime)
        with pytest.raises(TaskError):
            await vf.grade_submission(task, trace, runtime, on_score=judge)

    assert trace.rewards == {}
