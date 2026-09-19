"""Typed units, async stages, and explicit successful-result reuse.

Entrypoints create units, then call `Flow.run()` inside `async with Flow(...)`.
Run returns all unit states and a quiescent/draining reason; pipelines decide success.
Admission sees reserved executions in `flow.active`, including their executing stage.

Keyed work requires explicit `inputs`: core fingerprints only the key and those inputs.
Agent results use native traces; host work declares its output type with `fn(..., output=...)`.
Failed work is uncached. Reuse restores values, never sandbox side effects. Optional
GitArtifacts revisions preserve files independently of the unit's workflow HEAD.

Live routes and holds survive stage completion. Only notes present at stage start are
acknowledged; holds retain them. Data updates require a settled unit and its inspected HEAD.
The CLI exposes `inspect`, `steer`, and `drain`; pipeline entrypoints own launch and recovery.
"""

from verifiers.v1.flow.artifacts import GitArtifacts, Revision
from verifiers.v1.flow.calls import (
    AgentWork,
    CallFailed,
    Record,
    Result,
    Work,
    agent,
    fn,
    should_retry,
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
from verifiers.v1.flow.unit import Execution, Transition, Unit, UnitData, UnitState

__all__ = [
    "AgentWork",
    "CallFailed",
    "Ctx",
    "Execution",
    "Flow",
    "FlowConfig",
    "GitArtifacts",
    "Pipeline",
    "Record",
    "Result",
    "Revision",
    "RunResult",
    "Stopped",
    "Transition",
    "Unit",
    "UnitData",
    "UnitState",
    "Work",
    "agent",
    "drain_on_interrupt",
    "fn",
    "should_retry",
]
