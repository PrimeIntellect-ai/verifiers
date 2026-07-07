"""Demo 2: same-box agentic judging in a prime sandbox.

One box, two agents, sequential:
  1. the program provisions a sandbox from the solver's runtime policy
  2. the solver (bash harness, GLM-5.2) does real work in the box (writes a file)
  3. the program writes the solver's trace into the box (file-based judging)
  4. the judge (bash harness, gpt-5.4-mini) is placed into the SAME box, inspects both
     the world (the solver's artifacts) and the trace, and emits a JSON verdict
  5. the program parses the verdict from the judge's trace

Exercises: provision + borrowed runtime, remote (tunneled) interception, trace -> Task
chaining with lineage stamps, and the judge-sees-the-solver's-world placement pattern.
"""

import asyncio
import json
import re

import verifiers.v1 as vf

SOLVER_PROMPT = (
    "Compute the sum of all primes below 100 and write just that number to "
    "/app/answer.txt. Then reply with a one-sentence summary of what you did."
)

JUDGE_PROMPT = (
    "You are auditing another agent's work inside its own sandbox.\n"
    "- Its task: {solver_task!r}\n"
    "- Its full trajectory is at /app/evidence/trace.json\n"
    "- Its work product should be at /app/answer.txt\n"
    "Verify the work INDEPENDENTLY: recompute the expected result yourself (you have "
    "bash and python3), compare it against /app/answer.txt, and skim the trajectory "
    "for anything suspicious (e.g. hardcoding without computing).\n"
    "Reply with ONLY a JSON object: "
    '{{"verdict": "correct" | "incorrect", "answer_found": "<file contents>", '
    '"reasoning": "<one sentence>"}}'
)


def judge_task(solver_trace: vf.Trace) -> vf.Task:
    """A plain traces -> Task function: mint the judge's task from the solver's trace."""
    return vf.Task(
        idx=0,
        prompt=JUDGE_PROMPT.format(solver_task=solver_trace.task.prompt),
        sources=(solver_trace.id,),
        relation="judges",
    )


async def main() -> None:
    sandbox = vf.PrimeConfig(labels=["agent-programs-demo"])
    solver = vf.Agent("bash", vf.make_context("z-ai/glm-5.2"), sandbox)
    judge = vf.Agent("bash", vf.make_context("openai/gpt-5.4-mini"), sandbox)

    task = vf.Task(idx=0, prompt=SOLVER_PROMPT)
    async with solver.provision(task) as box:
        print(f"box up: {box.descriptor}")
        solver_trace = await solver.run(task, runtime=box)
        print(
            "solver stop:", solver_trace.stop_condition, "| error:", solver_trace.error
        )
        print("solver turns:", solver_trace.num_turns, "| usage:", solver_trace.usage)

        # File-based judging: the evidence goes into the box, not the prompt.
        await box.write(
            "/app/evidence/trace.json",
            json.dumps(solver_trace.to_record(), indent=2).encode(),
        )
        verdict_trace = await judge.run(judge_task(solver_trace), runtime=box)
        print(
            "judge stop:", verdict_trace.stop_condition, "| error:", verdict_trace.error
        )
        print("judge turns:", verdict_trace.num_turns, "| usage:", verdict_trace.usage)

    reply = (
        verdict_trace.assistant_messages[-1].content
        if verdict_trace.assistant_messages
        else ""
    )
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    verdict = json.loads(match.group()) if match else {"error": "no JSON", "raw": reply}
    print("verdict:", json.dumps(verdict, indent=2))
    print(
        "lineage: judge task sources =",
        verdict_trace.task.sources,
        "relation =",
        verdict_trace.task.relation,
    )
    print("solver box:", solver_trace.info["agent"]["runtime"])
    print("judge box: ", verdict_trace.info["agent"]["runtime"])


if __name__ == "__main__":
    asyncio.run(main())
