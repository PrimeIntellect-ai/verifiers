"""Agentic judging: a solver plays the task, then a judge verifies the work.

Two reusable envs share the grading protocol. `--env.id agentic-judge` provisions
a fresh box from the solver's runtime policy and restores only the task's collected
artifacts; `--env.id shared-agentic-judge` explicitly runs the judge in the
solver's box. The judge grades rubric criteria (`[env.task]`: policy prompt,
criteria file) and writes its verdicts to `/tmp/verdict.json`, with the solver's
full trace record uploaded at `/tmp/trace.json`. `finalize()` validates them
strictly onto the solver's trace — `judge/<name>` metrics plus a weighted-mean
`judge` reward, composed with the taskset's own rewards via `[env.score]`
(judge-only by default).

The environment id selects the runtime boundary; there is no mode boolean whose
value can disagree with the environment's security and artifact semantics.
"""

import json
import re
from pathlib import Path

from pydantic import FiniteFloat

import verifiers.v1 as vf
from verifiers.v1.judges.rubric import (
    Criterion,
    RubricVerdicts,
    load_criteria,
    score_verdicts,
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
    listing = "\n".join(
        f"- {c.name}: {c.text} (answer one of, worst to best: {', '.join(c.choices)})"
        for c in criteria
    )
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
The agent's raw trace record (JSON: messages, tool calls, and its `info`
artifacts) is written by the harness — not the agent — at `{TRACE_FILE}`. The
record can be very large — never dump it whole; peek selectively (list its
keys, then slice out specific fields with python or jq) and pull only what you
need. It is complete — it may also carry the task's own scores/metrics and
reference material (a gold answer, a reference solution, held-out tests). Those
are context, not your standard: recorded scores can be wrong and references can
be narrower than the task; do not over-index on how a reference solves it. Your
verdict is what YOU verified by execution."""

SHARED_WORKSPACE_NOTE = f"""\
## Your workspace

The graded agent worked in this sandbox. Its edits and any scoring side effects
are present. {_RECORD_NOTE}"""

ISOLATED_WORKSPACE_NOTE = f"""\
## Your workspace

This is a fresh sandbox built from the task's image. The task's published
artifacts were restored at their original paths; other changes made by the
graded agent are not present. The usual artifact location is
`{vf.ARTIFACTS_DIR}/` (for code tasks, typically a patch to read or apply), plus
any task-declared paths. {_RECORD_NOTE}"""

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
        artifacts: dict[str, bytes | None],
    ) -> None:
        super().__init__(data)
        self.files = files
        self.artifacts = artifacts

    @classmethod
    def from_trace(
        cls,
        solution: vf.Trace,
        config: "JudgeTaskConfig",
        share_runtime: bool = True,
    ) -> "JudgeTask":
        """Mint the judge's task from the solver's finished trace.

        `share_runtime` selects both the workspace note and artifact transport. In
        the solver's box the published artifacts are already on disk, so none
        travel; a fresh box gets the collected set, restored by `setup` at the
        paths they had.
        """
        solved = solution.task.data
        files = {TRACE_FILE: json.dumps(solution.to_record()).encode()}
        template = config.build_prompt()
        body = _render(template, prompt=solved.prompt_text)
        if "{prompt}" not in template:
            # A policy that doesn't place the task statement itself still needs it.
            body += "\n\n" + _render(TASK_SECTION, prompt=solved.prompt_text)
        workspace_note = (
            SHARED_WORKSPACE_NOTE if share_runtime else ISOLATED_WORKSPACE_NOTE
        )
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
                network_allow=solved.network_allow,
                network_block=solved.network_block,
            ),
            files=files,
            artifacts={} if share_runtime else solution.state.artifacts,
        )

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await vf.restore(runtime, self.artifacts)
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
    """The judge agent. Its runtime setting is ignored: both judging modes use the
    solver's resolved runtime policy, either by borrowing its box or provisioning a
    fresh equivalent one."""
    task: JudgeTaskConfig = JudgeTaskConfig()
    score: ScoreConfig = ScoreConfig()


class AgenticJudgeEnv(vf.Env[AgenticJudgeEnvConfig]):
    """Common agentic-judge protocol; subclasses choose the runtime boundary."""

    def __init__(self, config: AgenticJudgeEnvConfig) -> None:
        # Both modes use the solver's policy. Shared judging borrows that exact box;
        # isolated judging resolves the mirrored JudgeTask into a fresh equivalent.
        config.judge = config.judge.model_copy(
            update={"runtime": config.solver.runtime}
        )
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
        if isinstance(self.config.solver.runtime, vf.SubprocessConfig):
            raise TypeError(
                "agentic-judge requires the solver to run in a container, but it "
                "resolves to the subprocess runtime; use "
                "--env.solver.runtime.type docker or prime"
            )
        validate_pairing(judge, JudgeTask, self.config.judge.runtime)

    async def setup(self, agents: vf.Agents) -> None:
        # The judge grades the policy; its tokens are never training data.
        agents.judge.trainable = False

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        by_agent = {t.agent.name: t for t in episode.traces}
        solution = by_agent["solver"]
        if solution.termination is not None:
            return
        verdict = by_agent["judge"]
        if verdict.termination is not None:
            reward = verdict.reward
        else:
            verdicts = RubricVerdicts.model_validate(
                verdict.info.get("verdict")
            ).verdicts
            criteria = self.config.task.criteria()
            scores = score_verdicts(verdicts, criteria, "the rubric's")
            for criterion in criteria:
                solution.record_metric(
                    f"judge/{criterion.name}", scores[criterion.name]
                )
            total = sum(criterion.weight for criterion in criteria)
            reward = sum(c.weight * scores[c.name] for c in criteria) / total
        if self.config.score.task_weight != 1.0:
            for task_reward in solution.rewards.values():
                if task_reward is not None:
                    task_reward.weight *= self.config.score.task_weight
        solution.record_reward("judge", reward, weight=self.config.score.judge_weight)


class SharedAgenticJudgeEnv(AgenticJudgeEnv):
    """Judge the solver in its runtime, preserving the complete mutable workspace."""

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        async with agents.solver.provision(task) as box:
            solution = await agents.solver.run(task, runtime=box)
            if solution.termination is not None:
                return
            if not solution.ok:
                raise RuntimeError(
                    "the solver's rollout failed, so the judge never ran"
                )
            judge_task = JudgeTask.from_trace(solution, self.config.task)
            await agents.judge.run(judge_task, runtime=box)


class IsolatedAgenticJudgeEnv(AgenticJudgeEnv):
    """Judge only collected artifacts in a fresh box with the solver's policy."""

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        solution = await agents.solver.run(task)
        if solution.termination is not None:
            return
        if not solution.ok:
            raise RuntimeError("the solver's rollout failed, so the judge never ran")
        await agents.judge.run(
            JudgeTask.from_trace(solution, self.config.task, share_runtime=False)
        )
