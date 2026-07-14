"""llm-judge: any taskset, LLM-judged (built-in topology).

Two agents: a `solver` — any harness, running the seed tasks (`--topology.taskset.id`) — and
a `judge`, fixed to the in-process `direct` chat loop (an episode ≈ one API call), excluded
from training. `go` peels the judge's inputs off the finished solver episode — the seed
task's framing, its ground truth (an `answer` field, when the taskset carries one), and the
solver's *final message* (the last message of its trace's final branch: the committed
answer, not the reasoning path) — renders them into a `JudgeTask`, and the judge's verdict
lands on the *solver's* trace as a deferred reward. Grading composes with any taskset
without touching its code; the solver task's own `@reward`s still run.

The judge's model/client/sampling stay configurable (`--topology.judge.model <id>`); only
its harness is pinned down — a judge that runs a real harness and investigates the trace
with tools is the `agentic-judge` topology. (For a verdict baked into a task's own grading,
call `vf.Judge` from the task's `@reward`.)

    uv run eval --topology.id llm-judge --topology.taskset.id gsm8k-v1 -n 4
"""

import logging

from pydantic import SerializeAsAny, model_validator

from verifiers.v1.decorators import metric, reward, stop
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.harnesses.direct import DirectHarnessConfig
from verifiers.v1.task import DataT, Task, TaskData
from verifiers.v1.topology import (
    AgentConfig,
    AgentGraph,
    Agents,
    Topology,
    TopologyConfig,
)
from verifiers.v1.trace import Trace
from verifiers.v1.types import content_text

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are grading another agent's attempt at a task.

<task>
{task}
</task>
{reference}
<attempt>
{attempt}
</attempt>

Assess whether the attempt solves the task correctly and completely{against}. Think it \
through, then end your reply with a final line of exactly `SCORE: <n>`, where <n> is an \
integer from 0 (entirely wrong) to 10 (flawless)."""

REFERENCE_SECTION = """
<reference_answer>
{answer}
</reference_answer>
"""


def parse_score(text: str) -> float | None:
    """A `SCORE: <n>` verdict in `text`, normalized to 0..1 — the last such line
    (markdown `**`/backtick emphasis and a `/10` suffix tolerated), clamped to the 0-10
    scale. None when no verdict was committed (the caller decides what that means).
    The one verdict grammar shared by everything that asks for a 0-10 score line
    (`JudgeTask`, `agentic-judge`)."""
    for line in reversed(text.splitlines()):
        stripped = line.strip().strip("*`").strip()
        if not stripped.upper().startswith("SCORE:"):
            continue
        try:
            score = float(stripped.split(":", 1)[1].split("/")[0].strip())
        except ValueError:
            return None
        return min(max(score / 10.0, 0.0), 1.0)
    return None


class JudgeTask(Task[DataT]):
    """A grading assignment: `prompt` carries the rendered upstream task + ground truth +
    attempt, the class carries the stop and the verdict parser — question and rubric in one
    typed object, constructed in `go` from the solver's trace (the forward arrow)."""

    @classmethod
    def from_trace(cls, trace: Trace) -> "JudgeTask":
        """Render a finished episode into a grading task (the `Task.from_trace`
        convention): the upstream task's framing, its ground truth (an `answer` data
        field, when the row has one), and the attempt's final message — all read off
        the trace, which is self-describing (`trace.task.data`). Pure trace→task code."""
        data = trace.task.data
        answer = getattr(data, "answer", None)
        return cls(
            TaskData(
                idx=data.idx,
                prompt=JUDGE_PROMPT.format(
                    task=data.prompt_text,
                    reference=REFERENCE_SECTION.format(answer=answer)
                    if answer is not None
                    else "",
                    against=" (against the reference answer)"
                    if answer is not None
                    else "",
                    attempt=cls.attempt_text(trace),
                ),
            )
        )

    @staticmethod
    def attempt_text(trace: Trace) -> str:
        """What gets judged: the final message of the trace's final branch (its most recent
        full context), as plain text — the solver's committed answer, not its transcript."""
        branches = trace.branches
        messages = branches[-1].messages if branches else []
        return content_text(messages[-1].content) if messages else ""

    @staticmethod
    def parse_score(trace: Trace) -> float | None:
        """The judge's verdict — `parse_score` over its final reply."""
        return parse_score(trace.last_reply)

    @stop
    async def committed(self, trace: Trace) -> bool:
        """Stop once a verdict line has been emitted, so a judge doesn't keep
        investigating past its commitment."""
        return self.parse_score(trace) is not None


class JudgeAgentConfig(AgentConfig):
    """The judge: fixed to the in-process `direct` chat loop (an episode ≈ one API call),
    excluded from training. Its routing stays per-agent config (`--topology.judge.model
    <id>`, `--topology.judge.client...`); the harness is the one thing this topology locks —
    swapping it is what the `agentic-judge` topology is for."""

    harness: SerializeAsAny[HarnessConfig] = DirectHarnessConfig(id="direct")
    trainable: bool = False

    @model_validator(mode="before")
    @classmethod
    def _fixed_harness(cls, data):
        """Refuse a harness swap with a pointer to the right topology (runs before the
        base `_resolve_plugins`, so the id never narrows to a foreign harness config)."""
        if isinstance(data, dict):
            raw = data.get("harness")
            if isinstance(raw, HarnessConfig):
                raw = raw.model_dump()
            if isinstance(raw, dict) and raw.get("id") not in (None, "", "direct"):
                raise ValueError(
                    "the llm-judge judge harness is fixed (the in-process `direct` chat "
                    "loop); for a judge that runs a real harness over the solver's trace "
                    "use `--topology.id agentic-judge`, whose judge harness is configurable"
                )
        return data


class LLMJudgeConfig(TopologyConfig):
    solver: AgentConfig = AgentConfig()
    """The agent under evaluation — it runs the seed tasks, so `--topology.taskset.id` is
    the one required knob."""
    judge: JudgeAgentConfig = JudgeAgentConfig()
    weight: float = 1.0
    """Weight of the judge's verdict in the solver's reward (alongside whatever the
    solver task's own `@reward`s already recorded)."""


class LLMJudgeTopology(Topology[LLMJudgeConfig]):
    async def run(self, task: Task, agents: Agents) -> None:
        solver = await agents.solver.run(task)
        await agents.judge.run(JudgeTask.from_trace(solver), parents=[solver])

    @metric(agent="solver")
    async def judge_committed(self, trace: Trace, graph: AgentGraph) -> float:
        """Whether the judge actually committed to a verdict for this solver trace."""
        return float(self.verdict(trace, graph) is not None)

    @reward(agent="solver")
    async def judge(self, trace: Trace, graph: AgentGraph) -> float:
        """The judge's verdict, recorded on the *solver's* trace — a missing verdict
        (errored judge, or no SCORE line) scores 0, loudly. Weighted by `config.weight`
        (config-driven, so applied here rather than via the decorator's static weight)."""
        score = self.verdict(trace, graph)
        if score is None:
            judge = next(iter(graph.children(trace, agent="judge")), None)
            logger.warning(
                "judge returned no verdict for solver trace %s (%s)",
                trace.id,
                judge.error.type if judge and judge.error else "no SCORE line",
            )
        return (score or 0.0) * self.config.weight

    def verdict(self, trace: Trace, graph: AgentGraph) -> float | None:
        """This solver trace's parsed verdict: the score its judge child committed to."""
        judges = graph.children(trace, agent="judge")
        return JudgeTask.parse_score(judges[0]) if judges else None


__all__ = [
    "JudgeAgentConfig",
    "JudgeTask",
    "LLMJudgeConfig",
    "LLMJudgeTopology",
    "parse_score",
]
