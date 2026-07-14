"""AIME proposer-solver: make one AIME 2026 problem harder, then test a solver panel.

Each AIME seed produces one graph with a proposer root and ``num_solvers`` solver
children. The proposer can use bash and edit to derive and verify a harder problem,
then commits its question and answer through a structured tool. Solvers use the
tool-less direct harness and inherit AIME's math-verify correctness reward.
"""

import asyncio
from typing import cast

import verifiers.v1 as vf

from proposer_solver_v1.servers.submit import SubmissionState, SubmitToolset

AIME_TASKSET_ID = "aime26-v1"
AIME_INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"

PROPOSE_PROMPT = """You are given an AIME 2026 problem and its verified answer.

Create a new, harder version of the problem. It should remain a self-contained,
unambiguous AIME-style problem whose final answer is an integer from 0 through 999.
Preserve the mathematical character of the original problem rather than replacing it
with an unrelated problem.

Use your bash and edit tools as scratch space. Solve the new problem yourself and use
your tools to verify the submitted answer carefully. Correctness matters more than
novelty or difficulty.

Original problem:
{problem}

Original verified answer:
{answer}

When you are confident in both fields, call `propose_submit_problem` exactly once with:

- `question`: the complete new problem, ready to give directly to a solver;
- `answer`: its verified integer answer as a string, without `\\boxed{{}}` or prose.

Do not reveal the answer inside the question.
"""


def extract_boxed(text: str) -> str | None:
    """Return the content of the last balanced ``\boxed{...}``, if present."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    content_start = start + len("\\boxed{")
    index, depth = content_start, 1
    while index < len(text) and depth:
        depth += (text[index] == "{") - (text[index] == "}")
        index += 1
    if depth != 0:
        return None
    answer = text[content_start : index - 1].strip()
    return answer or None


class ProposeData(vf.TaskData):
    source_problem: str
    source_answer: str


class ProposeTask(vf.Task[ProposeData, SubmissionState]):
    """Ask the proposer to transform one AIME seed and commit a structured result."""

    tools = (SubmitToolset,)

    @classmethod
    def from_task(cls, task: vf.Task) -> "ProposeTask":
        prompt = task.data.prompt
        if not isinstance(prompt, str):
            raise ValueError("AIME proposer-solver requires a text seed problem")
        answer = getattr(task.data, "answer", None)
        if not isinstance(answer, str):
            raise ValueError(
                "AIME proposer-solver requires a seed task with a string answer"
            )
        problem = prompt.removeprefix(AIME_INSTRUCTION)
        return cls(
            ProposeData(
                idx=task.data.idx,
                prompt=PROPOSE_PROMPT.format(problem=problem, answer=answer),
                source_problem=problem,
                source_answer=answer,
            )
        )

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        state = cast(SubmissionState, trace.state)
        if state.submitted:
            trace.info["submission"] = {
                "question": state.question,
                "answer": state.answer,
            }

    @vf.stop
    async def submitted(self, trace: vf.Trace) -> bool:
        return cast(SubmissionState, trace.state).submitted

    @vf.reward(weight=0.1)
    async def parseable(self, trace: vf.Trace) -> float:
        submission = trace.info.get("submission")
        return float(
            isinstance(submission, dict)
            and bool(submission.get("question"))
            and bool(submission.get("answer"))
        )


class ProposerSolverConfig(vf.TopologyConfig):
    proposer: vf.AgentConfig = vf.AgentConfig()
    """The default harness: bash and edit tools plus the task's submission tool."""
    solver: vf.DirectAgentConfig = vf.DirectAgentConfig()
    """The in-process, tool-less direct harness."""
    num_solvers: int = 4


class ProposerSolverTopology(vf.Topology[ProposerSolverConfig]):
    def load_tasks(self) -> list[vf.Task]:
        """Load the fixed AIME 2026 seed taskset through the plugin boundary."""
        config_type = vf.taskset_config_type(AIME_TASKSET_ID)
        return vf.load_taskset(config_type(id=AIME_TASKSET_ID)).load()

    def complete(self, graph: vf.AgentGraph) -> bool:
        """Handled proposer/solver failures do not invalidate an otherwise finished graph."""
        return graph.error is None

    @staticmethod
    def solver_task(seed: vf.Task, proposer: vf.Trace) -> vf.Task:
        """Derive a new AIME task while retaining the seed task's scorer and config."""
        submission = proposer.info["submission"]
        data = seed.data.model_copy(
            update={
                "prompt": AIME_INSTRUCTION + submission["question"],
                "answer": submission["answer"],
            }
        )
        return type(seed)(data, seed.config)

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        propose_task = cast(vf.Task, ProposeTask.from_task(task))
        proposer = await agents.proposer.run(propose_task)
        if not isinstance(proposer.info.get("submission"), dict):
            return
        derived = self.solver_task(task, proposer)
        await asyncio.gather(
            *(
                agents.solver.run(derived, parents=[proposer])
                for _ in range(self.config.num_solvers)
            )
        )

    @vf.metric(agent="proposer")
    async def solve_rate(self, trace: vf.Trace, graph: vf.AgentGraph) -> float:
        """Correctness rate over successful solver children; errored solvers are excluded."""
        graded = [
            child
            for child in graph.children(trace, agent="solver")
            if not child.has_error
        ]
        if not graded:
            return 0.0
        return sum(child.rewards.get("correct", 0.0) for child in graded) / len(graded)

    @vf.reward(agent="solver", weight=0.1)
    async def parseable(self, trace: vf.Trace) -> float:
        return float(extract_boxed(trace.last_reply) is not None)

    @vf.reward(agent="proposer")
    async def difficulty(self, trace: vf.Trace) -> float:
        """Peak at a 50% solver pass rate and fall linearly to zero at 0% and 100%."""
        return 1.0 - 2.0 * abs(trace.metrics["solve_rate"] - 0.5)


__all__ = ["ProposerSolverTopology"]
