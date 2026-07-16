"""A proposer -> solver program.

  1. the proposer (gpt-5.4-mini) invents a problem with a known answer
  2. the program mints a typed ProposedTask from the proposer's trace (lineage stamped)
  3. the solver (GLM-5.2) attempts it n=4 times — fan-out is a plain asyncio.gather —
     each run scored by the task's own @reward
  4. the program reads the structure back off the lineage stamps

Exercises: cross-model chaining, typed Task/TaskData subclasses, task-scored agent
runs, concurrent fan-out on a pooled interception server. Note the Agent has no group
verb: each solver run scores its rollout on its own, and anything comparative across the
fan-out belongs to whoever gathered the traces — in training, prime-rl samples the group.
"""

import asyncio
import json
import re
import statistics

import verifiers.v1 as vf
from verifiers.v1.harnesses.default import DefaultHarness, DefaultHarnessConfig

PROPOSER_PROMPT = (
    "Invent one self-contained arithmetic word problem whose answer is a single "
    "integer. Make it require 3-4 reasoning steps (rates, remainders, or nested "
    "quantities). Reply with ONLY a JSON object: "
    '{"problem": "<the problem text>", "answer": <the integer>}'
)


class ProposedData(vf.TaskData):
    """A row minted by a proposer agent: the problem plus its claimed answer."""

    answer: str


class ProposedTask(vf.Task[ProposedData]):
    """Judgement for solver runs on a proposed task."""

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        numbers = re.findall(r"-?\d+", (trace.last_reply or "").replace(",", ""))
        return 1.0 if numbers and numbers[-1] == self.data.answer else 0.0


def proposed_task(proposer_trace: vf.Trace) -> ProposedTask:
    """traces -> Task: mint the solver task out of the proposer's final message."""
    reply = proposer_trace.last_reply
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if match is None:
        raise ValueError(f"proposer did not reply with a JSON object: {reply!r}")
    proposed = json.loads(match.group())
    return ProposedTask(
        ProposedData(
            idx=0,
            prompt=proposed["problem"]
            + "\n\nEnd your reply with just the final integer.",
            answer=str(proposed["answer"]),
            sources=(proposer_trace.id,),
            relation="solves",
        )
    )


async def main() -> None:
    harness = DefaultHarness(DefaultHarnessConfig())
    client = vf.resolve_client(vf.EvalClientConfig())  # one endpoint, one shared client
    proposer = vf.Agent(harness, "openai/gpt-5.4-mini", client)
    solver = vf.Agent(harness, "z-ai/glm-5.2", client)

    async with proposer, solver:
        proposer_trace = await proposer.run(
            vf.Task(vf.TaskData(idx=0, prompt=PROPOSER_PROMPT))
        )
        task = proposed_task(proposer_trace)
        print("proposed problem:", task.data.prompt_text[:200])
        print("claimed answer:  ", task.data.answer)

        traces = await asyncio.gather(*(solver.run(task) for _ in range(4)))

    # No topology object: the structure is already on the traces, via the stamps.
    for trace in traces:
        print(
            f"  solver {trace.id[:8]}: reward={trace.reward:.2f} rewards={trace.rewards} "
            f"turns={trace.num_turns} sources={trace.task.data.sources[0][:8]}({trace.task.data.relation})"
        )
    rewards = [t.reward for t in traces]
    print(
        f"solved by {proposer_trace.id[:8]}'s task: mean={statistics.mean(rewards):.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
