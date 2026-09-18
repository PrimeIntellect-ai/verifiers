"""Durable stages with typed workflow state and optional immutable Git artifacts.

    class TaskData(UnitData):
        revision: Revision | None = None

    async def review(ctx: Ctx[TaskData]) -> Transition[TaskData]:
        result = await ctx.call(fn(check, ctx.data.revision), key="check")
        return Transition.end("done", result)

    pipeline = Pipeline(stages={"review": review}, start="review", data=TaskData,
                        campaign_data=TaskData)

`UnitState` owns stage/status, controls, and notes; `data` is the pipeline's model.
`ctx.data` is a private copy of the stage's starting data. Return it (or `ctx.updated`)
in a Transition to publish it. Successful transitions acknowledge the starting notes;
holds retain them. Notes arriving during execution remain for the next stage.

`GitArtifacts(unit).write(base=revision, files=changes)` returns an immutable revision
without moving the workflow HEAD. Capture outputs before recording a successful call,
then adopt its returned revision through a transition. Call replay restores recorded
values, not sandbox side effects. Materialize a recorded revision into a fresh workspace.

`Unit.steer` and the CLI share audited controls. A live hold parks after the stage
finishes; data patches require no active stage and the expected workflow HEAD. Inspect
that boundary with `status <root> [unit] --json`; update using
`update <root> <unit> <patch.json> --expected <sha> [--stage <name>] [--status ready]`.
Run against the same root to resume; Flow exits when nothing is runnable.
"""

from verifiers.v1.flow.artifacts import GitArtifacts, Revision
from verifiers.v1.flow.calls import (
    AgentWork,
    CallFailed,
    Record,
    Result,
    Work,
    agent,
    command,
    fn,
    should_retry,
)
from verifiers.v1.flow.config import FlowConfig
from verifiers.v1.flow.flow import (
    CAMPAIGN,
    Ctx,
    Flow,
    Pipeline,
    Stopped,
    drain_on_interrupt,
    succeeded,
)
from verifiers.v1.flow.unit import Transition, Unit, UnitData, UnitState

__all__ = [
    "CAMPAIGN",
    "AgentWork",
    "CallFailed",
    "Ctx",
    "Flow",
    "FlowConfig",
    "GitArtifacts",
    "Pipeline",
    "Record",
    "Result",
    "Revision",
    "Stopped",
    "Transition",
    "Unit",
    "UnitData",
    "UnitState",
    "Work",
    "agent",
    "command",
    "drain_on_interrupt",
    "fn",
    "should_retry",
    "succeeded",
]
