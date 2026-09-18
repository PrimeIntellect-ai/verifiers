"""A small pipeline on the flow core: a campaign plans folders, each folder is built by a
seat in a box, reviewed, solved by several seats at once, judged, and banked.

    uv run python -m verifiers.v1.flow run examples.flow.flywheel:pipeline ./out @ cfg.toml
    uv run python -m verifiers.v1.flow status ./out
    uv run python -m verifiers.v1.flow release ./out folder-1 --note "the box was the problem"

A stage composes calls (seats, commands, functions, spreads) and returns where the unit
goes next. The unit's `state.json` is the only state; a stage rerun attaches to the calls it
already made. A partial spread is the stage's decision: here fewer than three answers holds
the unit for an operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import verifiers.v1 as vf
from verifiers.v1.flow import (
    Ctx,
    FlowConfig,
    Pipeline,
    Record,
    Transition,
    UnitData,
    Work,
    agent,
    command,
    fn,
)


class Cfg(FlowConfig):
    planner: vf.AgentConfig = vf.AgentConfig(harness={"id": "rlm"})
    builder: vf.AgentConfig = vf.AgentConfig(
        harness={"id": "rlm"}, runtime=vf.DockerConfig(image="python:3.12")
    )
    reviewer: vf.AgentConfig = vf.AgentConfig(harness={"id": "rlm"})
    solver: vf.AgentConfig = vf.AgentConfig(harness={"id": "rlm"})
    judge: vf.AgentConfig = vf.AgentConfig(harness={"id": "rlm"})
    folders: int = 3


class FolderData(UnitData):
    brief: str
    visits: int = 0


def task(prompt: str) -> vf.Task:
    return vf.Task(vf.TaskData(prompt=prompt))


def decision(trace: vf.Trace) -> str:
    return (
        (trace.last_reply or "").strip().split()[-1].lower()
        if trace.last_reply
        else "revise"
    )


async def plan(ctx: Ctx) -> Transition:
    trace = await ctx.call(
        agent("planner", task(f"Propose {ctx.config.folders} folders.")), key="plan"
    )
    for i, line in enumerate(
        (trace.last_reply or "").splitlines()[: ctx.config.folders]
    ):
        ctx.flow.create_task(f"folder-{i}", stage="build", data=FolderData(brief=line))
    return Transition.wait("planned")


@dataclass(frozen=True)
class BuildAndLint(Work[dict[str, Any]]):
    """Cache the complete box-dependent operation, never a trace without its files."""

    kind: ClassVar = "fn"
    brief: str

    def content(self, ctx: Ctx) -> list[Any]:
        return [
            "build-and-lint",
            self.brief,
            ctx.seat("builder").model_dump(mode="json"),
        ]

    async def execute(self, ctx: Ctx, name: str) -> dict[str, Any]:
        async with ctx.runtime("builder") as box:
            built = await ctx.call(agent("builder", task(self.brief), runtime=box))
            lint = await ctx.call(command(["ruff", "check", "."], runtime=box))
        return {"reply": built.last_reply, "lint": lint.exit_code}

    def dump(self, ctx: Ctx, value: dict[str, Any]) -> dict[str, Any]:
        return {"payload": value}

    def load(self, ctx: Ctx, record: Record) -> dict[str, Any]:
        return record.payload


async def build(ctx: Ctx[FolderData]) -> Transition[FolderData]:
    state = ctx.data
    built = await ctx.call(BuildAndLint(state.brief), key=f"build/{state.visits}")
    review = await ctx.call(
        agent("reviewer", task(f"Review:\n{built['reply']}\nlint: {built['lint']}")),
        key=f"review/{state.visits}",
    )
    if built["lint"] == 0 and decision(review) == "accept":
        return Transition.to("solve", "accepted", review.last_reply or "")
    if state.visits >= 2:
        return Transition.end("rejected", "the folder never passed review")
    return Transition.to(
        "build",
        "revise",
        review.last_reply or "",
        data=ctx.updated(visits=state.visits + 1),
    )


async def solve(ctx: Ctx[FolderData]) -> Transition[FolderData]:
    version = ctx.data.visits
    attempts = await ctx.spread(
        [agent("solver", task(ctx.data.brief)) for _ in range(4)],
        key=lambda i: f"solve/{version}/{i}",
    )
    answers = [r.value for r in attempts if r.ok and r.value is not None]
    if len(answers) < 3:
        return Transition.hold(
            f"{len(answers)}/4 solvers answered: {[r.error for r in attempts if not r.ok]}"
        )
    judged = await ctx.call(
        agent("judge", task("\n".join(a.last_reply or "" for a in answers))),
        key=f"judge/{version}",
    )
    bank = await ctx.call(fn(len, judged.last_reply or ""), key=f"bank/{version}")
    return Transition.end(
        "banked", f"{bank} chars", files={"judgment.md": judged.last_reply or ""}
    )


pipeline = Pipeline(
    {"plan": plan, "build": build, "solve": solve},
    start="plan",
    config=Cfg,
    data=FolderData,
)
