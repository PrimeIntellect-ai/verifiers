"""agentic-judge: any taskset, judged by a real agent over the uploaded trace (built-in topology).

The heavyweight sibling of `llm-judge`: same two agents, same verdict contract (the judge's
score lands on the *solver's* trace as a deferred reward), but the judge is a true agent in
its own runtime. Instead of rendering the attempt into a prompt, `go` serializes the solver's
*entire* trace — the task it was given (prompt plus any ground-truth fields), the complete
conversation, the rewards its own task recorded — and the judge task's `setup` hook uploads
it into the judge's freshly provisioned runtime before the harness runs. The judge then
investigates the file with its tools (the bash+edit `default` harness unless swapped) and
commits to a `SCORE: <n>` line.

Everything `llm-judge` locks down is configurable here: the judge's harness
(`--topology.judge.harness.id ...`, and where it runs via `--topology.judge.harness.runtime...`)
and its assignment (`--topology.judge.prompt`, well, `prompt` on the topology config).

    uv run eval --topology.id agentic-judge --topology.taskset.id gsm8k-v1 -n 4
"""

import json

from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, TaskData
from verifiers.v1.topology import AgentConfig, Agents, Topology
from verifiers.v1.topologies.llm_judge import (
    JudgeTask,
    LLMJudgeConfig,
    LLMJudgeTopology,
)
from verifiers.v1.trace import Trace

TRACE_PATH = "/tmp/solver_trace.json"
"""Where the solver's serialized trace lands in the judge's runtime."""

AGENTIC_JUDGE_PROMPT = """You are grading another agent's rollout on a task. Its full \
trace has been uploaded to `{path}` as JSON: the task it was given (`task` — the prompt, \
plus any ground-truth fields such as a reference answer), the complete conversation \
(`nodes`, one message per node), and any rewards its own task recorded (`rewards`).

Go look at `{path}` and assess whether the agent solved its task correctly and \
completely. Investigate however you like — the file can be large, so consider extracting \
the task and the agent's final messages first. When you have decided, end your reply with \
a final line of exactly `SCORE: <n>`, where <n> is an integer from 0 (entirely wrong) to \
10 (flawless)."""


class AgenticJudgeData(TaskData):
    """The grading assignment's wire half: the solver's entire trace, JSON-serialized
    (`Trace.to_record`) — persisted with the judge's own trace, so the record shows
    exactly what was judged."""

    trace_json: str


class AgenticJudgeTask(JudgeTask[AgenticJudgeData]):
    """A grading assignment over an uploaded trace: the data carries the solver's
    serialized trace, and `setup` writes it into the judge's runtime before the
    harness runs — so the file is there the moment the judge starts investigating. The
    verdict parser and the `committed` stop are inherited from `JudgeTask`."""

    @classmethod
    def from_trace(
        cls, trace: Trace, prompt: str = AGENTIC_JUDGE_PROMPT
    ) -> "AgenticJudgeTask":
        """Wrap a finished episode for agentic judging (the `Task.from_trace`
        convention): the assignment (`prompt`, with `{path}` resolved to where the
        trace will land) plus the trace itself."""
        return cls(
            AgenticJudgeData(
                idx=trace.task.data.idx,
                prompt=prompt.replace("{path}", TRACE_PATH),
                trace_json=json.dumps(trace.to_record()),
            )
        )

    async def setup(self, trace: Trace, runtime: Runtime) -> None:
        """Upload the judged trace into the judge's runtime."""
        await runtime.write(TRACE_PATH, self.data.trace_json.encode())


class AgenticJudgeAgentConfig(AgentConfig):
    """The agentic judge: a real agent, excluded from training. Runs the bash+edit
    `default` harness unless swapped (`--topology.judge.harness.id ...`) — unlike
    `llm-judge`, the harness here is the point of the topology, so it's configurable."""

    trainable: bool = False


class AgenticJudgeConfig(LLMJudgeConfig):
    judge: AgenticJudgeAgentConfig = AgenticJudgeAgentConfig()
    prompt: str = AGENTIC_JUDGE_PROMPT
    """The judge's assignment (`{path}` = where the trace was uploaded). A config field,
    so the rubric is tunable from the CLI/toml without a custom topology."""


class AgenticJudgeTopology(LLMJudgeTopology, Topology[AgenticJudgeConfig]):
    """Inherits the whole verdict contract from `LLMJudgeTopology` (`judge_committed`
    metric, weighted `judge` reward on the solver's trace); only the forward arrow —
    what the judge is handed — differs."""

    async def run(self, task: Task, agents: Agents) -> None:
        solver = await agents.solver.run(task)
        await agents.judge.run(
            AgenticJudgeTask.from_trace(solver, self.config.prompt),
            parents=[solver],
        )


__all__ = [
    "AgenticJudgeAgentConfig",
    "AgenticJudgeConfig",
    "AgenticJudgeData",
    "AgenticJudgeTask",
    "AgenticJudgeTopology",
]
