"""Shared-sandbox agentic style judging for software-engineering tasks.

The solver owns one runtime. After its task has been finalized and scored, a
non-trainable judge runs a second harness in that same live runtime. The judge sees the
working tree exactly as the solver left it and grades implementation quality without
receiving the solver's conversation.

    uv run --extra envs eval --topology.id swe-style-judge \
        --topology.taskset.id swebench-verified-v1 \
        --topology.taskset.use-prime-registry true \
        --topology.solver.harness.runtime.type prime \
        -m openai/gpt-5.5 -n 1 -r 1
"""

import logging

from pydantic import SerializeAsAny

from verifiers.v1.decorators import metric, reward
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.harnesses.default import DefaultHarnessConfig
from verifiers.v1.task import Task, TaskData
from verifiers.v1.topologies.llm_judge import JudgeTask
from verifiers.v1.topology import (
    AgentConfig,
    Agents,
    AgentGraph,
    Topology,
    TopologyConfig,
)
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

STYLE_JUDGE_PROMPT = """You are reviewing the implementation quality of another coding agent's work.

You are running in the exact same sandbox and working directory that the coding agent
used. The current filesystem is the artifact to review. Inspect the repository and its
working-tree changes directly with read-only commands such as `git status`, `git diff`,
and targeted file reads. Do not modify any files. Do not inspect hidden verifier data in
`/tests` or `/logs/verifier`.

The original task was:

<task>
{task}
</task>

Grade style and engineering quality, not functional correctness. Use this rubric:

The task verifier may temporarily replace or reset test files while scoring, so the
current working tree may not retain tests the solver added. Do not penalize missing test
changes or infer that the solver failed to write tests from their absence here.

- 0-2: scope discipline — focused changes with no unrelated churn.
- 0-3: clarity — readable, idiomatic code with sensible names and structure.
- 0-2: maintainability — robust handling without brittle hacks or needless complexity.
- 0-2: validation hygiene — appropriate tests or checks, without weakening existing tests.
- 0-1: repository hygiene — no generated junk, debug artifacts, or accidental edits.

Use at most three read-only tool calls. Even if the available evidence is incomplete,
you must finish the review and provide a score. Explain the most important evidence
briefly. End with a final line of exactly
`SCORE: <n>`, where <n> is an integer from 0 to 10.
"""


class StyleJudgeTask(JudgeTask):
    """A style-review assignment over the solver's still-live working tree."""

    @classmethod
    def from_trace(
        cls, trace: Trace, prompt: str = STYLE_JUDGE_PROMPT
    ) -> "StyleJudgeTask":
        return cls(
            TaskData(
                idx=trace.task.data.idx,
                name=trace.task.data.name,
                prompt=prompt.replace("{task}", trace.task.data.prompt_text),
            )
        )


class StyleJudgeAgentConfig(AgentConfig):
    """A tool-using, read-only-by-instruction judge excluded from training."""

    harness: SerializeAsAny[HarnessConfig] = DefaultHarnessConfig(
        id="default", edit=False
    )
    trainable: bool = False


class SWEStyleJudgeConfig(TopologyConfig):
    solver: AgentConfig = AgentConfig()
    judge: StyleJudgeAgentConfig = StyleJudgeAgentConfig()
    prompt: str = STYLE_JUDGE_PROMPT
    weight: float = 1.0


class SWEStyleJudgeTopology(Topology[SWEStyleJudgeConfig]):
    async def run(self, task: Task, agents: Agents) -> None:
        async with agents.solver.provision(task) as runtime:
            solution = await agents.solver.run(task, runtime=runtime)
            await agents.judge.run(
                StyleJudgeTask.from_trace(solution, self.config.prompt),
                parents=[solution],
                runtime=runtime,
            )

    @metric(agent="solver")
    async def style_committed(self, trace: Trace, graph: AgentGraph) -> float:
        return float(self.verdict(trace, graph) is not None)

    @reward(agent="solver")
    async def style(self, trace: Trace, graph: AgentGraph) -> float:
        score = self.verdict(trace, graph)
        if score is None:
            judge = next(iter(graph.children(trace, agent="judge")), None)
            logger.warning(
                "style judge returned no verdict for solver trace %s (%s)",
                trace.id,
                judge.error.type if judge and judge.error else "no SCORE line",
            )
        return (score or 0.0) * self.config.weight

    def verdict(self, trace: Trace, graph: AgentGraph) -> float | None:
        judges = graph.children(trace, agent="judge")
        return StyleJudgeTask.parse_score(judges[0]) if judges else None


__all__ = ["SWEStyleJudgeTopology"]
