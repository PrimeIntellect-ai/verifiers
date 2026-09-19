"""Durable stages with typed workflow state and optional immutable Git artifacts.

    class TaskData(UnitData):
        revision: Revision | None = None

    async def review(ctx: Ctx[TaskData]) -> Transition[TaskData]:
        result = await ctx.call(fn(check, ctx.data.revision), key="check")
        return Transition.end("done", result)

    pipeline = Pipeline(stages={"review": review})
    async with Flow(root, config, pipeline) as flow:
        flow.create_unit("task-1", stage="review", data=TaskData())
        result = await flow.run()

Every unit lives in `units/<id>` and carries its own data model. Flow creates none.
An entrypoint seeds units explicitly (the generic CLI calls `Pipeline.initialize`).
Repeated `create_unit` preserves existing work. `Pipeline.admit(unit, flow)` sees
`flow.active`, a read-only map of reserved executions: later candidates see earlier
admissions immediately. Each execution retains its starting stage after live routing.
There is no built-in coordinator, barrier, or success policy. `run()` returns
`RunResult(reason="quiescent" | "draining", units=...)`; pipelines interpret the states.

`Work.content(ctx)` declares reusable inputs; core fingerprints them with the call key.
`agent(..., inputs=...)` combines task data and resolved model/harness behavior with the
pipeline's explicit task configuration and external dependencies. Include grading when
returning a scored trace. Credentials, retries and run token/turn ceilings do not key the
resolved seat; request sampling does. To force fresh work, change its key or inputs.
`fn` uses its arguments unless given explicit `inputs`; `command` requires declared inputs
for its filesystem dependencies. Core neither infers dependencies nor restores side effects.

Stage executions, call invocations, attempts and native rollout retries have explicit IDs
in `transitions.jsonl`. Failed, unkeyed and cancelled work stays observable without a cache
record. Attachments create new invocation events referencing the original result's call
and execution. An unfinished start is incomplete evidence, never an inferred success.

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
    agent_inputs,
    command,
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
    "agent_inputs",
    "command",
    "drain_on_interrupt",
    "fn",
    "should_retry",
]
