"""A task-building flywheel as a flow: build → (lint, selftest, N solves) → review → decide.

    uv run python -m verifiers.v1.flow check examples.flow.flywheel:Flywheel
    uv run python -m verifiers.v1.flow run examples.flow.flywheel:Flywheel rows.jsonl runs/demo config.json

`rows.jsonl` holds one `{"brief": ...}` per line; `config.json` pins the model and,
for a real run, `{"type": "prime", "vm": true}` runtimes on the seats. Lint, selftest
and review run in the builder's live runtime; solvers get fresh runtimes and pull the
builder's tree in `SolveTask.setup` however the taskset chooses (git, tarball, image).
"""

from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.flow import (
    END,
    Flow,
    FlowConfig,
    OutcomeState,
    Upstream,
    agent,
    at_least,
    expand,
    fn,
    run,
)

WORKDIR = "/tmp/flywheel/work"


class FlywheelConfig(FlowConfig):
    builder: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "bash"}, runtime=vf.SubprocessConfig()
    )
    solver: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "bash"}, runtime=vf.SubprocessConfig()
    )
    reviewer: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "bash"}, runtime=vf.SubprocessConfig()
    )
    n_rollouts: int = 4


class Brief(vf.TaskData):
    brief: str = ""


class BuildTask(vf.Task[Brief, OutcomeState]):
    """The builder turns a brief into a task: README.md plus tests/test.sh."""


class SolveTask(vf.Task[vf.TaskData]):
    """A solver attempt in a fresh runtime, scored by the task's own hidden test."""

    @vf.reward
    async def tests_pass(self, runtime: vf.Runtime) -> float:
        result = await runtime.run(
            ["sh", "-c", f"cd {WORKDIR} && sh tests/test.sh"], {}
        )
        return float(result.exit_code == 0)


class ReviewTask(vf.Task[vf.TaskData, OutcomeState]):
    """The reviewer sees the check results and the solve rate, then decides."""


def build_task(up: Upstream) -> BuildTask:
    return BuildTask(
        Brief(
            workdir=WORKDIR,
            brief=up.row["brief"],
            prompt=(
                f"Working in {WORKDIR}, create a small coding task from this brief: {up.row['brief']}\n"
                "Write README.md (the task statement) and tests/test.sh (exits 0 iff solved). "
                "Call submit_outcome with `ready` when done, or `blocked` if the brief is unusable."
            ),
        )
    )


def solve_task(up: Upstream, attempt: int) -> SolveTask:
    prompt = f"Working in {WORKDIR}, solve the task in README.md. Attempt {attempt}."
    return SolveTask(vf.TaskData(workdir=WORKDIR, idx=attempt, prompt=prompt))


def review_task(up: Upstream) -> ReviewTask:
    return ReviewTask(
        vf.TaskData(
            workdir=WORKDIR,
            prompt=(
                f"Review the task in {WORKDIR}. Lint: {up.outcome('lint')}. Self-test: {up.outcome('selftest')}. "
                f"Solver pass rate: {up.aggregate['pass_rate']:.2f}. "
                "Call submit_outcome with accept, revise, or reject."
            ),
        )
    )


def aggregate(up: Upstream) -> dict:
    traces = up.solve
    return {"pass_rate": sum(t.reward > 0 for t in traces) / max(len(traces), 1)}


def decide(up: Upstream) -> Literal["ship", "rework"]:
    good = up.aggregate["pass_rate"] >= 0.5 and up.outcome("review") == "accept"
    return "ship" if good else "rework"


class Flywheel(Flow[FlywheelConfig]):
    build = agent(
        "builder",
        build_task,
        outcomes={"ready": ("lint", "selftest", "solve"), "blocked": END},
        max_visits=3,
        on_exhausted=END,
    )
    lint = run(
        [
            "sh",
            "-c",
            f"cd {WORKDIR} && test -s README.md && echo 'Outcome: ok' || echo 'Outcome: bad'",
        ],
        runtime="inherit:build",
        outcomes={"*": "review"},
    )
    selftest = run(
        [
            "sh",
            "-c",
            f"cd {WORKDIR} && test -f tests/test.sh && echo 'Outcome: ok' || echo 'Outcome: bad'",
        ],
        runtime="inherit:build",
        outcomes={"*": "review"},
    )
    solve = expand(
        "solver",
        solve_task,
        over=lambda up: range(up.config.n_rollouts),
        max_active=2,
        join=at_least(2),
        then="aggregate",
    )
    aggregate = fn(aggregate, then="review")
    review = agent(
        "reviewer",
        review_task,
        runtime="inherit:build",
        outcomes={"accept": "decide", "revise": "build", "reject": END},
    )
    decide = fn(decide, outcomes={"ship": END, "rework": "build"})
