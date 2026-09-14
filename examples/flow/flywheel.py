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
    FlowTask,
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


class BriefData(vf.TaskData):
    brief: str = ""


class BuildTask(FlowTask[BriefData]):
    """The builder turns a brief into a task: a README describing it and a `tests/` dir."""

    @classmethod
    def from_row(cls, up: Upstream) -> "BuildTask":
        return cls(
            BriefData(
                workdir=WORKDIR,
                brief=up.row["brief"],
                prompt=(
                    f"Working in {WORKDIR}, create a small coding task from this brief: {up.row['brief']}\n"
                    "Write README.md (the task statement) and tests/test.sh (exits 0 iff solved). "
                    "Call submit_outcome with `ready` when done, or `blocked` if the brief is unusable."
                ),
            )
        )


class SolveTask(FlowTask[BriefData]):
    """A solver attempt in a fresh runtime; scored by the task's own hidden test."""

    @classmethod
    def from_build(cls, up: Upstream, item: int) -> "SolveTask":
        return cls(
            BriefData(
                workdir=WORKDIR,
                idx=item,
                prompt=f"Working in {WORKDIR}, solve the task in README.md. Attempt {item}.",
            )
        )

    @vf.reward
    async def tests_pass(self, runtime: vf.Runtime) -> float:
        result = await runtime.run(
            ["sh", "-c", f"cd {WORKDIR} && sh tests/test.sh"], {}
        )
        return float(result.exit_code == 0)


class ReviewTask(FlowTask[BriefData]):
    """The reviewer sees the lint and self-test exit codes and the solve rate, then decides."""

    @classmethod
    def from_upstream(cls, up: Upstream) -> "ReviewTask":
        return cls(
            BriefData(
                workdir=WORKDIR,
                prompt=(
                    f"Review the task in {WORKDIR}. Lint exit code: {up.lint.exit_code}. "
                    f"Self-test exit code: {up.selftest.exit_code}. "
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
        BuildTask.from_row,
        outcomes={"ready": ("lint", "selftest", "solve"), "blocked": END},
        max_visits=3,
        on_exhausted=END,
    )
    lint = run(
        ["sh", "-c", f"cd {WORKDIR} && test -s README.md"],
        runtime="inherit:build",
        exit_codes={0: "ok", "*": "bad"},
        outcomes={"*": "review"},
    )
    selftest = run(
        ["sh", "-c", f"cd {WORKDIR} && test -f tests/test.sh"],
        runtime="inherit:build",
        exit_codes={0: "ok", "*": "bad"},
        outcomes={"*": "review"},
    )
    solve = expand(
        "solver",
        SolveTask.from_build,
        over=lambda up: range(up.config.n_rollouts),
        max_active=2,
        join=at_least(2),
        then="aggregate",
    )
    aggregate = fn(aggregate, then="review")
    review = agent(
        "reviewer",
        ReviewTask.from_upstream,
        runtime="inherit:build",
        outcomes={"accept": "decide", "revise": "build", "reject": END},
    )
    decide = fn(decide, outcomes={"ship": END, "rework": "build"})
