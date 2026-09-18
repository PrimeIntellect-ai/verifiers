"""Durable pipelines of agents over git.

    from verifiers.v1.flow import Ctx, Flow, FlowConfig, Pipeline, Transition, agent, fn

    async def review(ctx: Ctx) -> Transition:
        trace = await ctx.call(agent("reviewer", task(ctx.unit)), key="review")
        return Transition.to("build", trace.info["decision"], trace.last_reply or "")

    pipeline = Pipeline(stages={"plan": plan, "review": review, "build": build}, start="plan")

A unit is a git repository whose `state.json` names its stage; a stage is a function of a
`Ctx` that composes calls -- seats, commands, functions, spreads of them -- and returns a
`Transition`, committed as the unit's next state. A call with a key is recorded and found
again by a rerun; a stage that holds waits for an operator to call `Unit.steer` or the CLI.
"""

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
from verifiers.v1.flow.unit import Transition, Unit

__all__ = [
    "CAMPAIGN",
    "AgentWork",
    "CallFailed",
    "Ctx",
    "Flow",
    "FlowConfig",
    "Pipeline",
    "Record",
    "Result",
    "Stopped",
    "Transition",
    "Unit",
    "Work",
    "agent",
    "command",
    "drain_on_interrupt",
    "fn",
    "should_retry",
    "succeeded",
]
