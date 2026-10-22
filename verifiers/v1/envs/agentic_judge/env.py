"""A solver followed by an agentic judge in the selected verifier workspace.

The solver's ScoringConfig selects shared or isolated scoring. Task scoring and
agentic judging run in that same workspace and share its timeout/retry boundary.
The judge has its own non-training trace and writes rubric verdicts to a file.
"""

import copy
import json
import re
from pathlib import Path

from pydantic import FiniteFloat

import verifiers.v1 as vf
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.configs.scoring import resolve_scoring
from verifiers.v1.judges.rubric import (
    Criterion,
    RubricVerdicts,
    criterion_metrics,
    drop_reason,
    load_criteria,
    score_verdicts,
    weighted_score,
)
from verifiers.v1.utils.compile import validate_pairing

VERDICT_FILE = "/tmp/verdict.json"
TRACE_FILE = "/tmp/trace.json"

GRADE_PROMPT = """\
You are grading another agent's attempt at a task. Verify the work EMPIRICALLY:
reconstruct what the agent did from its trace and test it with real execution
in your sandbox — never take the trace's word for an outcome you can check."""

TASK_SECTION = """\
## The task the agent was given

{prompt}"""


SOLVED = Criterion(
    name="solved",
    text="The task is fully solved: what the task asked for is achieved, and "
    "you verified it with real execution.",
)


def _verdict_section(criteria: list[Criterion]) -> str:
    listing = "\n".join(c.render() for c in criteria)
    return f"""\
## Your verdict

Grade the attempt on these criteria:

{listing}

When you are done verifying, write your verdict as JSON to `{VERDICT_FILE}`:

    {{"verdicts": [{{"name": "<criterion name>", "reason": "<one sentence citing \
what you verified>", "verdict": "<answer>"}}, ...]}}

with one entry per criterion, using each criterion's exact name. For each, first
write the one-sentence reason grounded in what you actually verified, then set
verdict to exactly one of the options listed in parentheses after that
criterion."""


def _render(template: str, **fields: str) -> str:
    """Substitute documented placeholders in one pass over the original template —
    str.format would crash on any literal brace in a custom prompt, and sequential
    replaces would re-scan substituted values. An unknown placeholder stays as written."""
    pattern = re.compile(r"\{(" + "|".join(map(re.escape, fields)) + r")\}")
    return pattern.sub(lambda m: fields[m.group(1)], template)


_RECORD_NOTE = f"""\
The agent's observable trace record (JSON: messages, tool calls, and its `info`
artifacts) is written by the harness — not the agent — at `{TRACE_FILE}`. The
record can be very large — never dump it whole; peek selectively (list its
keys, then slice out specific fields with python or jq) and pull only what you
need. Hidden reasoning and opaque provider state are not included unless the run
explicitly opts in. The record may also carry the task's own scores/metrics and
reference material (a gold answer, a reference solution, held-out tests). Those are
context, not your standard: recorded scores can be wrong and references can be
narrower than the task; do not over-index on how a reference solves it. Your verdict
is what YOU verified by execution."""

SHARED_WORKSPACE_NOTE = f"""\
## Your workspace

The graded agent worked in this sandbox. Its edits and any scoring side effects
are present. {_RECORD_NOTE}"""

ISOLATED_WORKSPACE_NOTE = f"""\
## Your workspace

This verification sandbox was initialized from the selected verifier image.
The task's published artifacts were restored at their original paths; unpublished
solver changes are not present. The usual artifact location is
`{vf.ARTIFACTS_DIR}/`, plus task-declared paths. Task setup, verifier staging, and
task scoring have already run here: hidden tests, reference material, generated
files, and scoring side effects may also be present. Distinguish the submitted
artifacts from verifier-provided material; verify the submission empirically,
rather than treating a reference file or a recorded score as proof of success.
{_RECORD_NOTE}"""

HINT_SECTION = """\
## Hints

{hint}"""


class JudgeTask(vf.Task):
    """The judge's verdict task: the solver task's world mirrored onto the minted
    row, the trace record written (and any stale verdict removed) before the
    judge starts, verdict scraped off the live box after it exits. `NEEDS_CONTAINER`
    keeps `Agent.run`'s per-task backstop aligned with the judge's declared need."""

    NEEDS_CONTAINER = True

    def __init__(
        self,
        data: vf.TaskData,
        files: dict[str, bytes],
    ) -> None:
        super().__init__(data)
        self.files = files

    @classmethod
    def from_trace(
        cls,
        solution: vf.Trace,
        config: "JudgeTaskConfig",
        *,
        isolated: bool = False,
    ) -> "JudgeTask":
        """Build the verdict task for the already prepared scoring workspace."""
        solved = solution.task.data
        record = solution.to_record()
        # This snapshot is evidence for verification, not its final outcome.
        # stop() marks the agent loop completed before rollout scoring finishes.
        record.pop("ok", None)
        if not config.include_hidden_reasoning:
            for node in record["nodes"]:
                message = node["message"]
                message.pop("reasoning_content", None)
                message.pop("provider_state", None)
        files = {TRACE_FILE: json.dumps(record).encode()}
        template = config.build_prompt()
        body = _render(template, prompt=solved.prompt_text)
        if "{prompt}" not in template:
            # A policy that doesn't place the task statement itself still needs it.
            body += "\n\n" + _render(TASK_SECTION, prompt=solved.prompt_text)
        workspace_note = ISOLATED_WORKSPACE_NOTE if isolated else SHARED_WORKSPACE_NOTE
        sections = [body, _verdict_section(config.criteria()), workspace_note]
        if (hint := config.build_hint()) is not None:
            sections.insert(1, _render(HINT_SECTION, hint=hint))
        prompt = "\n\n".join(sections)
        return cls(
            vf.TaskData(
                idx=solved.idx,
                prompt=prompt,
                image=solved.image,
                workdir=solved.workdir,
                resources=solved.resources,
                network=solved.network,
            ),
            files=files,
        )

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # The solver had this box first: a pre-seeded verdict must never read as
        # the judge's own, and a file (or planted symlink) at an upload path must
        # never survive it — a symlinked TRACE_FILE would redirect the write onto
        # any file the solver chose.
        await runtime.run(["rm", "-f", VERDICT_FILE, *self.files], env={})
        for path, content in self.files.items():
            await runtime.write(path, content)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        """Scrape the verdict off the box while it's alive. A judge that wrote no
        file (or garbage) fails HERE — on the judge's own trace, the retryable
        unit — never silently."""
        try:
            raw = await runtime.read(VERDICT_FILE)
        except Exception as e:
            raise ValueError(
                f"the judge wrote no verdict to {VERDICT_FILE}; its final act must "
                'be writing {"verdicts": [{"name", "reason", "verdict"}, ...]} there'
            ) from e
        trace.info["verdict"] = RubricVerdicts.model_validate_json(raw).model_dump()


class TextFile(vf.BaseConfig):
    """An explicit file-backed text value for config formats without `Path` values."""

    path: Path


TextSource = str | Path | TextFile


class JudgeTaskConfig(vf.BaseConfig):
    """The judge's minted task: the grading policy and what lands in its box."""

    prompt: TextSource | None = None
    """Grading policy: a string is inline text; a `Path` in Python or
    `{ path = "policy.md" }` in config reads a file. Replaces only the policy
    body — the verdict contract and workspace note are always appended. May
    reference `{prompt}` (the solver task's prompt); if it doesn't, the task
    statement is appended after."""
    hint: TextSource | None = None
    """Optional hints, with the same explicit inline/file forms as `prompt`,
    injected as their own section: task-family pointers into the trace or box —
    e.g. for math, where the reference answer lives in the record; for SWE, to
    diff the repo or read `info.patch`."""
    rubric: Path | None = None
    """Criteria the judge grades against: a `.toml`/`.json` file with a
    `criteria` list — the plugged rubric judge's format, so the same rubric
    files work for both. None grades the single built-in `solved` criterion."""
    include_hidden_reasoning: bool = False
    """Include assistant reasoning and opaque provider state in `/tmp/trace.json`.
    Disabled by default so the judge grades only observable agent behavior."""

    @staticmethod
    def _resolve(value: TextSource) -> str:
        if isinstance(value, str):
            return value
        path = value if isinstance(value, Path) else value.path
        return path.read_text(encoding="utf-8")

    def build_prompt(self) -> str:
        if self.prompt is None:
            return GRADE_PROMPT + "\n\n" + TASK_SECTION
        return self._resolve(self.prompt)

    def build_hint(self) -> str | None:
        return self._resolve(self.hint) if self.hint is not None else None

    def criteria(self) -> list[Criterion]:
        if self.rubric is None:
            return [SOLVED]
        return load_criteria(self.rubric)


class ScoreConfig(vf.BaseConfig):
    """How the judge's verdict composes with the taskset's own rewards on the
    solver's trace. Judge-only by default."""

    task_weight: FiniteFloat = 0.0
    """Scale applied to the taskset's own rewards; 1 keeps them next to the verdict."""
    judge_weight: FiniteFloat = 1.0
    """Weight of the judge's verdict in the solver's reward."""


class AgenticJudgeEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    """The solver agent. Its runtime must be a container:
    `--env.solver.runtime.type docker|prime`."""
    judge: vf.AgentConfig = vf.AgentConfig()
    """The judge program; it borrows the workspace selected by solver.scoring."""
    task: JudgeTaskConfig = JudgeTaskConfig()
    score: ScoreConfig = ScoreConfig()


class AgenticJudgeEnv(vf.Env[AgenticJudgeEnvConfig]):
    """Run an empirical judge during the solver's verification phase."""

    def __init__(self, config: AgenticJudgeEnvConfig) -> None:
        super().__init__(config)
        self._check_agents()
        # A missing policy file or a malformed rubric fails here, not mid-episode.
        config.task.build_prompt()
        config.task.build_hint()
        config.task.criteria()

    def _check_agents(self) -> None:
        """The judge executes real code, never on the host — refuse an impossible
        pairing at construction, not after burning a full solver run."""
        judge = self._harnesses["judge"]
        if not judge.EXECUTES_CODE:
            raise ValueError(
                "agentic-judge requires a judge harness that can execute code, but "
                f"harness {judge.config.id!r} is a tool-less chat loop — a verdict "
                "that needs no execution is a plugged judge "
                "(--env.taskset.task.judges), not an agent."
            )
        if self.config.judge.scoring.model_fields_set:
            raise ValueError(
                "configure scoring placement on solver.scoring, not judge.scoring"
            )

    async def setup(self, agents: vf.Agents) -> None:
        # The judge grades the policy; its tokens are never training data.
        agents.judge.trainable = False

    def _record_verdict(self, solution: vf.Trace, verdict: vf.Trace) -> None:
        verdicts = RubricVerdicts.model_validate(verdict.info.get("verdict")).verdicts
        criteria = self.config.task.criteria()
        scores = score_verdicts(verdicts, criteria, "the rubric's")
        solution.record_metrics(criterion_metrics("judge", verdicts, criteria, scores))
        if self.config.score.task_weight != 1.0:
            for reward in solution.rewards.values():
                if reward is not None:
                    reward.weight *= self.config.score.task_weight
        if (reason := drop_reason("judge", verdicts, criteria)) is not None:
            solution.drop(reason)
            return
        solution.record_reward(
            "judge",
            weighted_score(scores, criteria),
            weight=self.config.score.judge_weight,
        )

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        policy = resolve_scoring(task.scoring, self.config.solver.scoring)
        workspace = (
            policy.runtime if policy.mode == "isolated" else None
        ) or self.config.solver.runtime
        if isinstance(workspace, vf.SubprocessConfig):
            raise TypeError("agentic judging requires a container scoring workspace")
        validate_pairing(self._harnesses["judge"], JudgeTask, workspace)

        async def judge(solution: vf.Trace, runtime: vf.Runtime) -> None:
            isolated = policy.mode == "isolated"
            judge_task = JudgeTask.from_trace(
                solution,
                self.config.task,
                isolated=isolated,
            )
            # The workspace already has the verifier's image and network policy.
            # Do not reapply the solver's requirements to an overridden verifier.
            updates = {
                "image": runtime.config.image,
                "workdir": runtime.config.workdir,
            }
            if isinstance(runtime.config, NetworkPolicyConfig):
                updates["network"] = NetworkPolicyConfig(
                    allow=runtime.config.allow, block=runtime.config.block
                )
            judge_task = judge_task.with_data(**updates)
            # A per-invocation view retains episode accounting and concurrency while
            # binding placement to the already provisioned scoring workspace.
            judge_agent = copy.copy(agents.judge)
            judge_agent.runtime_config = runtime.config
            judge_agent.config = judge_agent.config.model_copy(
                update={"runtime": runtime.config}
            )
            verdict = await judge_agent.run(judge_task, runtime=runtime)
            if not verdict.ok:
                raise RuntimeError(
                    "agentic judge failed: "
                    + "; ".join(error.message for error in verdict.errors)
                )
            self._record_verdict(solution, verdict)

        await agents.solver.run(task, on_score=judge)
