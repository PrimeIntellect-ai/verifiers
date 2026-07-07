"""A proposer -> solver program.

  1. the proposer (gpt-5.4-mini) invents a problem with a known answer
  2. the program mints a typed ProposedTask from the proposer's trace (lineage stamped)
  3. the solver (GLM-5.2) attempts it n=4 times — fan-out is a plain asyncio.gather —
     each run scored by a taskset @reward
  4. the program reads the structure back off the lineage stamps

Exercises: cross-model chaining, typed Task subclass, taskset-scored agent runs,
concurrent fan-out on a pooled interception server. Note there is no group concept
here: verifiers scores rollouts; grouping (and anything comparative across a group)
belongs to the consumer — in training, prime-rl samples the group.
"""

import asyncio
import json
import re
import statistics

import verifiers.v1 as vf

PROPOSER_PROMPT = (
    "Invent one self-contained arithmetic word problem whose answer is a single "
    "integer. Make it require 3-4 reasoning steps (rates, remainders, or nested "
    "quantities). Reply with ONLY a JSON object: "
    '{"problem": "<the problem text>", "answer": <the integer>}'
)


class ProposedTask(vf.Task):
    """A task minted by a proposer agent: the problem plus its claimed answer."""

    answer: str


class SolveTaskset(vf.Taskset[ProposedTask, vf.TasksetConfig]):
    """Judgement for solver runs on a proposed task."""

    @vf.reward
    async def correct(self, task: ProposedTask, trace: vf.Trace) -> float:
        reply = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        numbers = re.findall(r"-?\d+", (reply or "").replace(",", ""))
        return 1.0 if numbers and numbers[-1] == task.answer else 0.0


def proposed_task(proposer_trace: vf.Trace) -> ProposedTask:
    """traces -> Task: mint the solver task out of the proposer's final message."""
    reply = (
        proposer_trace.assistant_messages[-1].content
        if proposer_trace.assistant_messages
        else ""
    )
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    proposed = json.loads(match.group())
    return ProposedTask(
        idx=0,
        prompt=proposed["problem"] + "\n\nEnd your reply with just the final integer.",
        answer=str(proposed["answer"]),
        sources=(proposer_trace.id,),
        relation="solves",
    )


async def main() -> None:
    proposer = vf.Agent("default", vf.make_context("openai/gpt-5.4-mini"))
    solver = vf.Agent("default", vf.make_context("z-ai/glm-5.2"))
    taskset = SolveTaskset(vf.TasksetConfig())

    async with proposer, solver:
        proposer_trace = await proposer.run(vf.Task(idx=0, prompt=PROPOSER_PROMPT))
        task = proposed_task(proposer_trace)
        print("proposed problem:", task.prompt[:200])
        print("claimed answer:  ", task.answer)

        traces = await asyncio.gather(
            *(solver.run(task, taskset=taskset) for _ in range(4))
        )

    # No topology object: the structure is already on the traces, via the stamps.
    for trace in traces:
        print(
            f"  solver {trace.id[:8]}: reward={trace.reward:.2f} rewards={trace.rewards} "
            f"turns={trace.num_turns} sources={trace.task.sources[0][:8]}({trace.task.relation})"
        )
    rewards = [t.reward for t in traces]
    print(
        f"solved by {proposer_trace.id[:8]}'s task: mean={statistics.mean(rewards):.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
