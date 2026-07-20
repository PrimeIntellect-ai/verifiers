"""proposer_solver: a proposer invents a verified math problem; n solvers race it.

The task-generation recipe env. The "proposer" seat plays the dataset (a topic
seed), uses its tools to CONSTRUCT and verify a hard integer-answer problem, and
ends with a JSON contract. The env mints a typed `SolveTask` from that trace and
fans it out to `--env.n` independent runs of the "solver" seat. Each solve is judged by the minted
task's own reward (exact final integer, against the proposer's verified answer);
the proposer is judged by what its problem DOES to the solvers — `learnability`
peaks when half of them crack it (4p(1-p), the automatic-curriculum signal) — plus
a `solve_rate` metric.

Seats are deliberately heterogeneous: point the proposer at a code-running harness
in a real sandbox and keep the solvers on a cheap tool-less chat loop —

    uv run eval proposer-solver-v1 -n 4 \
      --env.proposer.harness.id codex --env.proposer.harness.runtime.type prime \
      --env.solver.harness.id null

Train-side, the seats flip independently per run (`--env.solver.trainable false`
trains only on proposer rollouts; both default trainable, late-bound to the run's
model).
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
    """The proposer's seat task: no per-trace judgement — the proposer is judged
    cross-agent, by what its problem does to the solvers."""


class SolveData(vf.TaskData):
    answer: str
    """The proposer's verified answer — the minted task's ground truth."""


class SolveTask(vf.Task[SolveData]):
    @classmethod
    def from_trace(cls, proposer: vf.Trace) -> "SolveTask":
        """trace -> Task: the proposer's JSON contract becomes the solvers' task.
        Off-contract output raises — the env-rollout fails (retryable) instead of
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
        answer = proposed["answer"]
        if not isinstance(answer, int) or isinstance(answer, bool):
            raise ValueError(
                f"proposer's contract answer is not a JSON integer: {answer!r}"
            )
        return cls(
            SolveData(
                idx=proposer.task.data.idx,
                prompt=str(proposed["problem"])
                + "\n\nEnd your reply with just the final integer.",
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


class ProposerSolverEnv(vf.Environment[ProposerSolverEnvConfig]):
    def roles(self):
        # The proposer plays the dataset seeds; the solvers play env-minted plain
        # problems — bare model actors, so any taskset needs don't apply to them.
        return {
            "proposer": vf.Role(self.config.proposer),
            "solver": vf.Role(self.config.solver, mcp=False, container=False),
        }

    async def rollout(self, task, agents):
        proposed = await agents["proposer"].run(task)
        solve_task = SolveTask.from_trace(proposed)
        solves = await asyncio.gather(
            *(agents["solver"].run(solve_task) for _ in range(self.config.n))
        )
        return {"proposer": proposed, "solver": list(solves)}

    @staticmethod
    def _solve_rate(traces: list[vf.Trace]) -> float:
        solves = [t for t in traces if t.role == "solver"]
        if not solves:
            return 0.0
        return sum(t.rewards.get("correct", 0.0) for t in solves) / len(solves)

    @vf.reward(role="proposer")
    async def learnability(self, trace, traces):
        """The curriculum signal: 1.0 when half the solvers crack the problem, 0
        when it's trivial or impossible for them (4p(1-p))."""
        rate = self._solve_rate(traces)
        return 4.0 * rate * (1.0 - rate)

    @vf.metric(role="proposer")
    async def solve_rate(self, trace, traces):
        return self._solve_rate(traces)


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
