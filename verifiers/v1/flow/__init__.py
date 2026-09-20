"""Typed units, async stages, and explicit successful-result reuse.

Entrypoints create units, then call `Flow.run()` inside `async with Flow(...)`.
Run returns all unit states and a quiescent/draining reason; pipelines decide success.
Stages can annotate `Ctx[MyData, MyConfig] -> Transition[MyData]`; edit `ctx.data`
and publish it with `data=ctx.data`. Different stages may use different data models.
Admission sees reserved executions in `flow.active`, including their executing stage.
When accepting work from another execution, use `ctx.link_from(source_id, label="...")`.
This records provenance, not scheduling or successful completion.

Keyed work requires explicit `inputs`: core fingerprints only the key and those inputs.
Agent results use native traces; host work declares its output type with `fn(..., output=...)`.
Failed work is uncached. Reuse restores values, never sandbox side effects. Optional
GitArtifacts revisions preserve files independently of the unit's workflow HEAD.

Live routes and holds survive stage completion. Only notes present at stage start are
acknowledged; holds retain them. Data updates require a settled unit and its inspected HEAD.
The CLI exposes `inspect`, `steer`, and `drain`; pipeline entrypoints own launch and recovery.
"""

from verifiers.v1.flow.artifacts import ArtifactRevision, GitArtifacts
from verifiers.v1.flow.calls import (
    AgentWork,
    CallFailed,
    Failure,
    Record,
    Result,
    Success,
    Work,
    fn,
)
from verifiers.v1.flow.config import FlowConfig
from verifiers.v1.flow.flow import (
    Ctx,
    Flow,
    Pipeline,
    RunResult,
    Stopped,
    drain_on_interrupt,
)
from verifiers.v1.flow.stats import FlowStats, Stats, summarize
from verifiers.v1.flow.unit import (
    Execution,
    Transition,
    Unit,
    UnitData,
    UnitInspection,
    UnitState,
)

__all__ = [
    "AgentWork",
    "ArtifactRevision",
    "CallFailed",
    "Ctx",
    "Execution",
    "Failure",
    "Flow",
    "FlowConfig",
    "FlowStats",
    "GitArtifacts",
    "Pipeline",
    "Record",
    "Result",
    "RunResult",
    "Stats",
    "Stopped",
    "Success",
    "Transition",
    "Unit",
    "UnitData",
    "UnitInspection",
    "UnitState",
    "Work",
    "drain_on_interrupt",
    "fn",
    "summarize",
]
