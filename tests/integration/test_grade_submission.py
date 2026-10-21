"""`grade_submission(on_score=...)`: a model-free grading's post-scoring callback runs
in the fresh verifier, and only an attempt it finishes commits its scores; an agent
judge runs there through `judge_solution`."""

import json
import shutil
from typing import ClassVar

import pytest

import verifiers.v1 as vf
from verifiers.v1.envs.agentic_judge import (
    Criterion,
    JudgeTaskConfig,
    judge_solution,
)
from verifiers.v1.errors import TaskError
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.runtimes import ProgramResult, provision_runtime

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
        agent=vf.AgentInfo(
            config=vf.AgentConfig(
                model="test/fake-model", runtime=runtime.backend.config
            )
        ),
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


class InspectingHarness(NullHarness):
    """A judge program that inspects the box it runs in and writes its verdict:
    `answer` is yes when /app/answer holds 42. `seen` records what it found."""

    SUPPORTS_MCP = False
    SUPPORTS_TOOL_INTERCEPTION = False
    seen: ClassVar[list[dict]] = []
    write_verdict = True

    async def setup(self, runtime: vf.Runtime) -> None:
        return None

    async def launch(
        self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, **kwargs
    ) -> ProgramResult:
        answer = await runtime.run(["cat", "/app/answer"], {})
        solver_only = await runtime.run(["test", "-e", "/var/tmp/solver-only"], {})
        record = await runtime.run(["test", "-s", "/tmp/trace.json"], {})
        type(self).seen.append(
            {
                "answer": answer.stdout.strip(),
                "solver_only": solver_only.exit_code == 0,
                "record": record.exit_code == 0,
            }
        )
        if type(self).write_verdict:
            verdict = "yes" if answer.stdout.strip() == "42" else "no"
            body = {
                "verdicts": [
                    {"name": "answer", "reason": "read it", "verdict": verdict}
                ]
            }
            await runtime.write("/tmp/verdict.json", json.dumps(body).encode())
        return ProgramResult(exit_code=0, stdout="", stderr="")


def judge_agent() -> vf.Agent:
    agent = vf.Agent(vf.AgentConfig(model="test/fake-model", harness={"id": "null"}))
    agent.harness = InspectingHarness(agent.harness.config)
    return agent


JUDGE_TASK = JudgeTaskConfig(
    criteria=[Criterion(name="answer", text="/app/answer holds 42.")]
)


async def test_judge_solution_grades_in_the_verifier_through_the_judges_verdict():
    """`judge_solution` from an `on_score`: the judge runs in the fresh verifier with
    the solver's record, and its verdict file gives each criterion's score."""
    InspectingHarness.seen, InspectingHarness.write_verdict = [], True
    task = answer_task()

    async def judge(trace: vf.Trace, runtime: vf.Runtime) -> None:
        scores, verdict = await judge_solution(
            judge_agent(), trace, runtime, JUDGE_TASK, isolated=True
        )
        assert verdict.info["verdict"]["verdicts"][0]["verdict"] == "yes"
        trace.record_reward("judge", scores["answer"])

    config = vf.DockerConfig(image=IMAGE, workdir="/app")
    async with provision_runtime(config) as runtime:
        trace = await solved_submission(task, runtime)
        await vf.grade_submission(task, trace, runtime, on_score=judge)

    assert InspectingHarness.seen == [
        {"answer": "42", "solver_only": False, "record": True}
    ]
    assert trace.rewards["judge"].value == 1.0


async def test_a_judge_that_writes_no_verdict_scores_nothing():
    InspectingHarness.seen, InspectingHarness.write_verdict = [], False
    task = answer_task()

    async def judge(trace: vf.Trace, runtime: vf.Runtime) -> None:
        await judge_solution(judge_agent(), trace, runtime, JUDGE_TASK, isolated=True)

    config = vf.DockerConfig(image=IMAGE, workdir="/app")
    async with provision_runtime(config) as runtime:
        trace = await solved_submission(task, runtime)
        with pytest.raises(TaskError):
            await vf.grade_submission(task, trace, runtime, on_score=judge)

    assert len(InspectingHarness.seen) == 1
    assert trace.rewards == {}
