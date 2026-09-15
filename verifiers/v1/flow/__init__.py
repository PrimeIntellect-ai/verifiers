"""Durable pipelines over verifiers primitives.

    from verifiers.v1.flow import Ctx, FlowConfig, Run, agent, command, fn

    async def pipeline(ctx: Ctx, row: dict) -> dict:
        review = await ctx.step("review", agent("reviewer", review_task(row)))
        async with ctx.runtime("builder") as box:
            build = await ctx.step("build", agent("builder", build_task(row), runtime=box))
            lint = await ctx.step("lint", command(["ruff", "check", "."], runtime=box))
        solves = await ctx.spread("solve", [agent("solver", t) for t in tasks])
        return await ctx.step("bank", fn(bank, review, build, lint, solves))

    results = await Run(run_dir, config).run(pipeline, rows)

A flow is a function; routing, loops and joins are Python. Every step is recorded
and a re-run against the same directory attaches to what finished.
"""

from verifiers.v1.flow.config import FlowConfig
from verifiers.v1.flow.ledger import Ledger, StepRecord
from verifiers.v1.flow.pools import Pools
from verifiers.v1.flow.run import (
    Ctx,
    RowResult,
    RowState,
    Run,
    RunStatus,
    StepFailed,
    Stopped,
)
from verifiers.v1.flow.work import (
    AgentWork,
    CommandWork,
    FnWork,
    Work,
    WorkKind,
    agent,
    command,
    fn,
)

__all__ = [
    "AgentWork",
    "CommandWork",
    "Ctx",
    "FlowConfig",
    "FnWork",
    "Ledger",
    "Pools",
    "RowResult",
    "RowState",
    "Run",
    "RunStatus",
    "StepFailed",
    "StepRecord",
    "Stopped",
    "Work",
    "WorkKind",
    "agent",
    "command",
    "fn",
]
