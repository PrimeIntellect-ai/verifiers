"""A submit tool: the agent ends its run by handing in a typed result.

The pattern data pipelines keep rebuilding (task QC, task authoring, review and judge
seats), assembled from Verifiers primitives — there is no submit-tool class to inherit:

- The payload is a pydantic model. The tool's JSON schema comes from it, and a call
  that does not validate comes back to the agent as a tool error inside the same
  rollout. Checks a schema cannot say (cross-field rules, files the result names)
  raise `ValueError` in the tool body and come back the same way.
- The tool records the result on the rollout's state; a `@vf.stop` ends the run
  once it is there, so the agent does not keep working after it handed in.
- An agent that stops without submitting gets a bounded number of reminders in the
  same interaction (`run_until_submitted`).
- `finalize` tells the two failed endings apart. An agent timeout STOPS the rollout
  (`trace.is_timeout`, `stop_condition == "agent_timeout"`) rather than failing it,
  so `finalize` still runs and can keep partial work; raising here fails the call.

The toolset lives in its own module, `submit_tools.py`: a tool server is started as
`python -m <its module>`, so that module's `__main__` must serve it (`SubmitTools.run()`)
and the module must be importable where the server starts — here, through PYTHONPATH.

`PRIME_API_KEY` in the environment, then:
PYTHONPATH=examples uv run python examples/submit_tool.py
"""

import asyncio

from submit_tools import SubmitState, SubmitTools

import verifiers.v1 as vf


class Undecided(Exception):
    """The agent's run ended without a submission."""


class OutOfTime(Undecided):
    """The agent's time budget ran out before it submitted."""


class ReviewTask(vf.Task[vf.TaskData, SubmitState]):
    @classmethod
    def toolsets(cls, config: vf.TaskConfig) -> list[vf.Toolset]:
        return [SubmitTools(vf.ToolsetConfig())]

    @vf.stop
    def submitted(self, trace: vf.Trace) -> bool:
        return trace.state.submission is not None

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        submission = trace.state.submission
        if submission is None:
            if trace.is_timeout:
                raise OutOfTime("the time budget ran out before submit()")
            raise Undecided("the run ended without calling submit()")
        trace.info["submission"] = submission.model_dump()


REMINDER = (
    "You stopped without calling `submit`. Your result only counts once you call it: "
    "finish what is left, then call `submit` with your verdict."
)


async def run_until_submitted(
    agent: vf.Agent, task: vf.Task, reminders: int = 2
) -> vf.Trace:
    """Run `task`, reminding an agent that stops without submitting, at most
    `reminders` times. A run that ended (a limit, a timeout, a failure, the
    `submitted` stop) gets no reminder."""
    async with agent.interaction(task) as ix:
        segment = await ix.turn()
        for _ in range(reminders):
            trace = ix.trace
            if (
                segment.terminated
                or trace.state.submission is not None
                or trace.stop_condition is not None
                or trace.last_error is not None
            ):
                break
            segment = await ix.turn(REMINDER)
    return ix.trace


async def main() -> None:
    agent = vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2"))
    task = ReviewTask(
        vf.TaskData(
            prompt="Is 2**61 - 1 prime? Submit `accept` if it is, `reject` if not.",
            timeout=vf.TaskTimeout(agent=600),
        )
    )
    async with agent:
        trace = await run_until_submitted(agent, task)
    print("stop:", trace.stop_condition, "error:", trace.last_error)
    print("submission:", trace.info.get("submission"))


if __name__ == "__main__":
    asyncio.run(main())
