"""Durable pipelines over verifiers primitives.

    from verifiers.v1.flow import Ctx, FlowConfig, Run, agent, command, fn

    async def pipeline(ctx: Ctx, row: dict) -> dict:
        review = await ctx.step("review", agent("reviewer", review_task(row)))
        async with ctx.runtime("builder") as box:
            build = await ctx.step("build", agent("builder", build_task(row), runtime=box))
            lint = await ctx.step("lint", command(["ruff", "check", "."], runtime=box))
        solves = await ctx.spread("solve", [agent("solver", t) for t in tasks])
        return await ctx.step("bank", fn(bank, review, build, lint, solves))

    async with Run(run_dir, config) as run:
        results = await run.run(pipeline, rows)

A flow is a function; routing, loops and joins are Python. Every step is recorded
and a re-run against the same directory attaches to what finished.
"""

from verifiers.v1.flow.config import FlowConfig
from verifiers.v1.flow.run import (
    Ctx,
    RowResult,
    Run,
    StepFailed,
    Stopped,
    drain_on_interrupt,
)
from verifiers.v1.flow.work import agent, command, fn

__all__ = [
    "Ctx",
    "FlowConfig",
    "RowResult",
    "Run",
    "StepFailed",
    "Stopped",
    "agent",
    "command",
    "drain_on_interrupt",
    "fn",
]
