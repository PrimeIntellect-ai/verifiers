"""proposer-solver-v1 — a proposer invents a code-checkable puzzle; n solvers reason it out.

The asymmetric-capability shape, one self-contained package:

  - **One seed, self-minted ground truth**: `load_tasks` returns a single `ProposeTask`.
    The proposer must invent a question that a short deterministic script can *solve*, but
    which reads as a natural-language puzzle a human (or a tool-less solver) must *reason*
    through. It commits the proposal by calling the `submit_question` tool.
  - **Tasks carry their behavior**: `ProposeTask` exposes the submit `Toolset` (`load_tools`),
    peels the committed submission off `trace.state` into the persisted `trace.info` in
    `finalize`, and judges its own episode (`well_formed`: the submission exists AND its
    ground-truth code runs cleanly on the given input). `SolverTask` carries that ground
    truth *and* the `correct` reward — question and verifier in one typed object, minted in
    `go`.
  - **Agents are pure routing, but asymmetric**: `proposer` is pinned to the `null` harness
    (a chat loop *with* MCP tools — it must call `submit_question`); `solver` is pinned to
    the tool-less in-process `direct` loop (it genuinely cannot execute code, so it has to
    reason). No task-side anything on either agent.
  - **`go` is control flow**: propose (the seed), read the committed submission off the
    proposer trace (the forward arrow, pure host-side), mint one `SolverTask`, fan
    `num_solvers` solver episodes out over it.
  - **Cross-agent judgement is declared**: the proposer's `difficulty` reward derives from
    its solvers' pass rate after the instance completes — peaked at 50%, so puzzles that are
    neither trivial nor impossible score highest.
"""

import asyncio
import re
from typing import ClassVar

import verifiers.v1 as vf

from proposer_solver_v1.servers.submit import SubmissionState, SubmitToolset

PROPOSE_PROMPT = """You are inventing a puzzle for a reasoning contest.

Correctness matters more than creativity. Your first response should be a single
`propose_submit_question` tool call; do not write a prose answer before it.

Design ONE compact problem that satisfies BOTH of these:
  1. It is solved by a complete stdlib-only Python script that reads one argument from
     `sys.argv[1]` and prints exactly one answer.
  2. It is also solvable by careful mental arithmetic from the natural-language question.

Use small integers, short strings, or simple counting. Avoid ambiguity, outside facts,
randomness, hidden state, TODOs, placeholders, or incomplete code.

A safe pattern is a letter-value puzzle: choose a short uppercase word, ask for the sum of
its letter positions (A=1, B=2, ..., Z=26), and submit code equivalent to:

import sys
s = sys.argv[1].strip()
print(sum(ord(ch) - ord("A") + 1 for ch in s))

Commit your puzzle by calling the `propose_submit_question` tool exactly once, with:
  - `code`: the complete ground-truth script,
  - `input`: the concrete input string,
  - `question`: the self-contained natural-language question for the solver.

Do not print the answer yourself."""

SOLVE_PROMPT = """Solve this puzzle. You have no code execution — reason it out carefully.

{question}

Show your reasoning, then give the final answer alone on the last line as \
`ANSWER: <value>`."""


def parse_number(text: str) -> str | None:
    """Canonicalize a numeric answer string (`$1,200.` → `1200`) for exact comparison;
    None when it isn't a number."""
    cleaned = text.strip().strip("`").lstrip("$").replace(",", "").rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return str(int(value)) if value == int(value) else str(value)


def parse_answer(trace: vf.Trace) -> str | None:
    """The value of the last `ANSWER: <value>` line of the trace's final reply, or None.
    Markdown emphasis around the line (`**ANSWER: 42**`, `` `ANSWER: 42` ``) is stripped."""
    for line in reversed(trace.last_reply.splitlines()):
        text = line.strip().strip("*`").strip()
        if text.upper().startswith("ANSWER:"):
            return text.split(":", 1)[1].strip().strip("*`").strip()
    return None


def _answer_tokens(text: str) -> list[str]:
    """An answer string as comparable tokens: split on commas/whitespace (brackets
    stripped), each token canonicalized numerically when it parses as a number."""
    parts = [p for p in re.split(r"[,\s]+", text.strip().strip("[]()")) if p]
    return [parse_number(p) or p.lower() for p in parts]


def answers_match(expected: str, got: str) -> bool:
    """Compare two answer strings: numerically when both parse whole as numbers (`1200`
    matches `$1,200.` — commas as thousands separators), else as token sequences,
    numerically per token — so `6,2,8,4,0,6` matches `6 2 8 4 0 6` (separator style is
    not the solver's problem) but not `6 2 8 4 0 7`."""
    e_num, g_num = parse_number(expected), parse_number(got)
    if e_num is not None and g_num is not None:
        return e_num == g_num
    return _answer_tokens(expected) == _answer_tokens(got)


UV_SCRIPT_HEADER = "# /// script\n# dependencies = []\n# ///\n"
"""Prepended to the submitted ground-truth code before execution: `runtime.run_uv_script`
runs PEP 723 scripts (it refuses a bare file), while the proposer submits plain,
stdlib-only Python — so the empty-deps header is supplied host-side."""


async def run_ground_truth(runtime: vf.Runtime, code: str, input: str) -> str | None:
    """Run the submitted ground-truth script on `input` in the runtime; the expected
    answer (the last stdout line), or None when the script doesn't run cleanly."""
    result = await runtime.run_uv_script(UV_SCRIPT_HEADER + code, args=[input])
    if result.exit_code != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip().splitlines()[-1].strip()


class ProposeTask(vf.Task):
    """The single question-writing assignment (the seed). Exposes the submit tool, peels the
    committed proposal off `trace.state` into the persisted `trace.info["submission"]`
    (`finalize`), and judges its own episode with a small `well_formed` reward. Whether the
    puzzle was any *good* arrives later, from the solvers (the topology's `difficulty`)."""

    STATE: ClassVar[type[vf.State]] = SubmissionState

    def load_tools(self) -> list[vf.Toolset]:
        return [SubmitToolset(vf.ToolsetConfig())]

    async def finalize(self, trace: vf.Trace) -> None:
        """Copy the committed submission from the transient `trace.state` into the persisted
        `trace.info` — readable by `go` in-process and by anyone reading `results.jsonl`."""
        if trace.state.submitted:
            trace.info["submission"] = {
                "code": trace.state.code,
                "input": trace.state.input,
                "question": trace.state.question,
            }

    @vf.stop
    async def submitted(self, trace: vf.Trace) -> bool:
        """Stop the episode once the proposal is committed — the tool write syncs to
        `trace.state`, so the proposer doesn't idle to the harness timeout after its
        single required tool call."""
        return trace.state.submitted

    @vf.reward(weight=0.1)
    async def well_formed(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """A submission exists AND its ground-truth code runs cleanly on the given input."""
        submission = trace.info.get("submission")
        if not submission or not submission.get("code"):
            return 0.0
        expected = await run_ground_truth(
            runtime, submission["code"], submission.get("input", "")
        )
        return float(expected is not None)


class SolverTask(vf.Task):
    """A generated puzzle — ground truth (`code` + `input`) and verifier in one typed object,
    minted in `go`. Serialized with each solver trace, so the record shows exactly what was
    asked and how it was graded."""

    code: str
    """The proposer's deterministic ground-truth script (reads `sys.argv[1]`, prints answer)."""
    input: str
    """The concrete input string the ground-truth script receives."""

    @vf.reward
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """Run the ground-truth `code` on `input` for the expected answer, parse the solver's
        final `ANSWER:` line, and compare. Missing line or mismatch → 0."""
        got = parse_answer(trace)
        if got is None:
            return 0.0
        expected = await run_ground_truth(runtime, self.code, self.input)
        if expected is None:
            return 0.0  # broken ground truth — don't credit or blame the solver
        return float(answers_match(expected, got))


class ProposerSolverConfig(vf.TopologyConfig):
    proposer: vf.NullAgentConfig = vf.NullAgentConfig()
    """The proposer needs the `null` chat loop — a program WITH the task's MCP tools, so
    it can call `submit_question` (`direct` sets `SUPPORTS_MCP=False` and would refuse
    the pairing)."""
    solver: vf.DirectAgentConfig = vf.DirectAgentConfig()
    """The solver is tool-less in-process `direct` — it genuinely cannot execute code,
    so it must reason the puzzle out."""
    num_solvers: int = 4
    """Solver episodes per proposed puzzle (the fan-out width)."""


class ProposerSolverTopology(vf.Topology[ProposerSolverConfig]):
    def load_tasks(self) -> list[vf.Task]:
        """Self-seeding: a single propose assignment (no `--topology.taskset.id` needed)."""
        return [ProposeTask(idx=0, prompt=PROPOSE_PROMPT)]

    async def go(self, task: vf.Task, run: vf.TopologyRun) -> None:
        """Control flow only: propose, read the committed submission off the trace (pure
        host-side), then fan the solvers out over the derived puzzle."""
        proposer = await run.agent("proposer").run(task)
        submission = proposer.info.get("submission")
        if not isinstance(submission, dict):
            return  # proposer never committed — `well_formed` scored it; nothing to solve
        question = submission.get("question")
        code = submission.get("code")
        input = submission.get("input")
        if not question or not code or input is None:
            return  # malformed submission — nothing solvable
        derived = SolverTask(
            idx=task.idx,
            prompt=SOLVE_PROMPT.format(question=question),
            code=code,
            input=input,
        )
        solver = run.agent("solver")
        await asyncio.gather(
            *(
                solver.run(derived, parents=[proposer])
                for _ in range(self.config.num_solvers)
            )
        )

    @vf.metric(agent="proposer")
    async def solve_rate(self, trace: vf.Trace, graph: vf.AgentGraph) -> float:
        """This proposal's pass rate across its solver children (errored episodes are
        excluded from the denominator rather than counted as wrong)."""
        graded = [t for t in graph.children(trace, agent="solver") if not t.has_error]
        return (
            sum(t.rewards.get("correct", 0.0) for t in graded) / len(graded)
            if graded
            else 0.0
        )

    @vf.reward(agent="proposer")
    async def difficulty(self, trace: vf.Trace) -> float:
        """Difficulty-targeted, off the `solve_rate` metric (metrics run before rewards):
        peaks at a 50% solve rate, 0 at trivial (100%) or impossible (0%)."""
        return 1.0 - 2.0 * abs(trace.metrics["solve_rate"] - 0.5)


__all__ = ["ProposerSolverTopology"]
