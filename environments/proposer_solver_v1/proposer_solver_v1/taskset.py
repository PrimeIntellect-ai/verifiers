"""proposer-solver: a proposer invents a verified math problem; n solvers race it.

The task-generation recipe env, as an example package. The "proposer" plays the
dataset (a topic seed), constructs and verifies a hard integer-answer problem with
its tools, and ends with a JSON contract; the env mints a typed `SolveTask` from
that trace and fans it to `--env.n` independent runs of the "solver". Each solve
is judged by the minted task's own reward; the proposer is judged by what its
problem DOES to the solvers — `learnability` peaks when half of them crack it
(4p(1-p), the automatic-curriculum signal).

    uv run eval proposer-solver-v1 -n 4 \
      --env.proposer.harness.id codex --env.proposer.harness.runtime.type prime \
      --env.solver.harness.id null
"""

import asyncio
import json
import re

from pydantic import Field

import verifiers.v1 as vf

PROPOSE = (
    "Invent ONE self-contained, genuinely hard {topic} word problem whose final "
    "answer is a single integer. Use your tools: write and run code to construct "
    "the numbers and to VERIFY the answer end-to-end before you commit to it. Do "
    "not reveal the solution path in the problem text. Your FINAL reply must end "
    "with exactly one JSON object on its own line: "
    '{{"problem": "<problem text>", "answer": <integer>}}'
)


class SeedData(vf.TaskData):
    topic: str


class ProposeTask(vf.Task[SeedData]):
    """The proposer's task: no per-trace judgement — the proposer is judged
    cross-trace, by what its problem does to the solvers."""


class SolveData(vf.TaskData):
    answer: str
    """The proposer's verified answer — the minted task's ground truth."""


class SolveTask(vf.Task[SolveData]):
    @classmethod
    def from_trace(cls, proposer: vf.Trace) -> "SolveTask":
        """trace -> Task: the proposer's JSON contract becomes the solvers' task.
        Off-contract output raises — the episode fails (retryable) instead of
        scoring solvers against garbage."""
        reply = proposer.last_reply or ""
        greedy = re.search(r"\{.*\}", reply, re.DOTALL)
        chunks = [line.strip() for line in reversed(reply.splitlines())]
        if greedy is not None:
            chunks.append(greedy.group())
        proposed = None
        for chunk in chunks:
            if not (chunk.startswith("{") and chunk.endswith("}")):
                continue
            try:
                parsed = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and {"problem", "answer"} <= parsed.keys():
                proposed = parsed
                break
        if proposed is None:
            raise ValueError(
                f"proposer did not end with the JSON contract: ...{reply[-300:]!r}"
            )
        problem, answer = proposed["problem"], proposed["answer"]
        if not isinstance(problem, str):
            raise ValueError(
                f"proposer's contract problem is not a string: {problem!r}"
            )
        if not isinstance(answer, int) or isinstance(answer, bool):
            raise ValueError(
                f"proposer's contract answer is not a JSON integer: {answer!r}"
            )
        return cls(
            SolveData(
                idx=proposer.task.data.idx,
                prompt=problem + "\n\nEnd your reply with just the final integer.",
                answer=str(answer),
            )
        )

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        numbers = re.findall(r"-?\d+", (trace.last_reply or "").replace(",", ""))
        return 1.0 if numbers and numbers[-1] == self.data.answer else 0.0


class ProposerSolverEnvConfig(vf.EnvConfig):
    proposer: vf.AgentConfig = vf.AgentConfig()
    solver: vf.AgentConfig = vf.AgentConfig()
    n: int = Field(4, ge=1)
    """Independent solver runs per proposed problem."""
    train_proposer: bool = True
    """Whether proposer rollouts are training data (`--env.train_proposer false`
    trains only on solver rollouts)."""
    train_solver: bool = True
    """Whether solver rollouts are training data (`--env.train_solver false`
    trains only on proposer rollouts)."""


class ProposerSolverEnv(vf.Env[ProposerSolverEnvConfig]):
    def setup(self, agents):
        # Both agents CAN train (same underlying policy); which one does this run
        # is this env's explicit choice to expose.
        agents.proposer.trainable = self.config.train_proposer
        agents.solver.trainable = self.config.train_solver

    async def run(self, task, agents):
        proposed = await agents.proposer.run(task)
        solve_task = SolveTask.from_trace(proposed)
        async with asyncio.TaskGroup() as group:
            for _ in range(self.config.n):
                group.create_task(agents.solver.run(solve_task))

    async def finalize(self, task, traces):
        """The curriculum signal: `learnability` rewards the proposer where half
        the solvers crack the problem, 0 where it's trivial or impossible for
        them (4p(1-p))."""
        solves = [t for t in traces if t.agent.name == "solver"]
        rate = (
            sum(t.rewards.get("correct", 0.0) for t in solves) / len(solves)
            if solves
            else 0.0
        )
        for trace in traces:
            if trace.agent.name == "proposer":
                trace.record_reward("learnability", 4.0 * rate * (1.0 - rate))
                trace.record_metric("solve_rate", rate)


class ProposerSolverTaskset(vf.Taskset[ProposeTask, vf.TasksetConfig]):
    TOPICS = [
        "rates and mixtures",
        "number theory",
        "combinatorics",
        "geometry",
        "probability",
        "modular arithmetic",
    ]

    def load(self) -> list[ProposeTask]:
        return [
            ProposeTask(
                SeedData(
                    idx=i,
                    name=topic.replace(" ", "-"),
                    prompt=PROPOSE.format(topic=topic),
                    topic=topic,
                ),
                self.config.task,
            )
            for i, topic in enumerate(self.TOPICS)
        ]
