"""agentic-judge: a solver plays the task, a code-executing judge verifies the work.

A reusable env (`--env.id agentic-judge` over any taskset). The solver plays the
task in a container provisioned from its runtime policy; the judge then grades
rubric criteria (`[env.task]`: policy prompt, criteria file) and writes its
verdicts to `/tmp/verdict.json`, with the solver's full trace record uploaded at
`/tmp/trace.json`. `finalize()` validates them strictly onto the solver's trace —
`judge/<name>` metrics plus a weighted-mean `judge` reward, composed with the
taskset's own rewards via `[env.score]` (judge-only by default).

`--env.topology` decides where the judge stands. Under `shared` (the default) it
plays in the box the agent worked in, seeing that environment directly and every
seam in it. Under `isolated` it gets its own box from the same image, holding only
what the task declared as artifacts, so nothing the agent did to its environment
can reach the grader — worth opting into once a taskset publishes its evidence
somewhere that travels.
"""

import json
import math
import re
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

import verifiers.v1 as vf
from verifiers.v1.artifacts import CONVENTION_DIR
from verifiers.v1.types import StrictBaseModel

VERDICT_FILE = "/tmp/verdict.json"
TRACE_FILE = "/tmp/trace.json"

GRADE_PROMPT = """\
You are grading another agent's attempt at a task. Verify the work EMPIRICALLY:
reconstruct what the agent did from its trace and test it with real execution
in your sandbox — never take the trace's word for an outcome you can check."""

TASK_SECTION = """\
## The task the agent was given

{prompt}"""


class Criterion(StrictBaseModel):
    """One rubric criterion — the plugged rubric judge's format, mirrored so the
    same `criteria` files grade both judges."""

    name: str
    """Key for the criterion's metric (`judge/<name>`)."""
    text: str
    weight: float = 1.0
    """The criterion's share of the reward."""
    choices: list[str] = Field(default_factory=lambda: ["no", "yes"])
    """Allowed answers, ordered **worst → best**: the first scores 0.0, the last 1.0, the rest
    evenly spaced by rank. Default `["no", "yes"]` is a binary check. Needs >= 2, no duplicates."""

    @field_validator("choices")
    @classmethod
    def _check_choices(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError(f"`choices` needs at least two options, got {v}")
        if len(set(v)) != len(v):
            raise ValueError(f"`choices` has duplicate options: {v}")
        return v


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
The agent's raw
trace record (JSON: messages, tool calls, and its `info` artifacts) is uploaded
at `{TRACE_FILE}`. The record can be very large — never dump it whole; peek
selectively (list its keys, then slice out specific fields with python or jq)
and pull only what you need. It is complete — it may also carry the task's own
scores/metrics and reference material (a gold answer, a reference solution,
held-out tests). Those are context, not your standard: recorded scores can be
wrong and references can be narrower than the task; do not over-index on how a
reference solves it. Your verdict is what YOU verified by execution."""

SHARED_SANDBOX_NOTE = f"""\
## Your workspace

Your sandbox is the SAME box the graded agent worked in, in the state the agent
left it — its edits (and any scoring side effects) are applied. {_RECORD_NOTE}"""

ISOLATED_SANDBOX_NOTE = f"""\
## Your workspace

Your sandbox is a FRESH box, built from the same image the graded agent started
from — so it holds the task's original state, NOT the state the agent left. The
agent's environment is gone; you cannot inspect it, and nothing it changed is
here except what the task declared as an artifact. Those artifacts have been
restored at their original paths (a patch under `{CONVENTION_DIR}/` is the usual
one for code tasks), so to see the agent's work you generally have to apply or
read them rather than looking at the working tree. If something you need to
check was never declared as an artifact, say so in your reason rather than
assuming its absence means the agent failed. {_RECORD_NOTE}"""

HINT_SECTION = """\
## Hints

{hint}"""


class JudgeTask(vf.Task):
    """The judge's verdict task: the solver task's world mirrored onto the minted
    row, the trace record written (and any stale verdict removed) before the
    judge starts, verdict scraped off the live box after it exits. `NEEDS_CONTAINER`
    keeps `Agent.run`'s per-task backstop aligned with the judge's declared need."""

    NEEDS_CONTAINER = True

    def __init__(self, data: vf.TaskData, files: dict[str, bytes]) -> None:
        super().__init__(data)
        self.files = files

    @classmethod
    def from_trace(
        cls,
        solution: vf.Trace,
        config: "JudgeTaskConfig",
        topology: str = "shared",
    ) -> "JudgeTask":
        """Mint the judge's task from the solver's finished trace.

        `topology` selects the workspace note. It has to match how the judge is
        actually placed: the note is the judge's only account of what its box
        contains, and a judge told it is standing in the agent's workspace when it
        is standing in a fresh one will read an unmodified tree as a failed attempt.
        """
        solved = solution.task.data
        files = {TRACE_FILE: json.dumps(solution.to_record()).encode()}
        template = config.build_prompt()
        body = _render(template, prompt=solved.prompt_text)
        if "{prompt}" not in template:
            # A policy that doesn't place the task statement itself still needs it.
            body += "\n\n" + _render(TASK_SECTION, prompt=solved.prompt_text)
        note = SHARED_SANDBOX_NOTE if topology == "shared" else ISOLATED_SANDBOX_NOTE
        sections = [body, _verdict_section(config.criteria()), note]
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
        trace.info["verdict"] = json.loads(raw)


class JudgeTaskConfig(vf.BaseConfig):
    """The judge's minted task: the grading policy and what lands in its box."""

    prompt: Path | str | None = None
    """Grading-policy override: inline text, or a policy file (a value ending in
    `.md`/`.txt` is read from disk). Replaces only the policy body — the verdict
    contract and workspace note are always appended, so a custom policy cannot
    break verdict scraping. May reference `{prompt}` (the solver task's prompt);
    if it doesn't, the task statement is appended after the policy."""
    hint: Path | str | None = None
    """Optional hints injected as their own section (inline text or a `.md`/
    `.txt` file): task-family pointers into the trace or box — e.g. for math,
    where the reference answer lives in the record; for SWE, to diff the repo
    or read `info.patch`."""
    rubric: Path | None = None
    """Criteria the judge grades against: a `.toml`/`.json` file with a
    `criteria` list — the plugged rubric judge's format, so the same rubric
    files work for both. None grades the single built-in `solved` criterion."""

    @staticmethod
    def _resolve(value: Path | str) -> str:
        path = Path(value)
        if isinstance(value, Path) or path.suffix in (".md", ".txt"):
            return path.read_text(encoding="utf-8")
        return str(value)

    def build_prompt(self) -> str:
        if self.prompt is None:
            return GRADE_PROMPT + "\n\n" + TASK_SECTION
        return self._resolve(self.prompt)

    def build_hint(self) -> str | None:
        return self._resolve(self.hint) if self.hint is not None else None

    def criteria(self) -> list[Criterion]:
        if self.rubric is None:
            return [SOLVED]
        text = self.rubric.read_text(encoding="utf-8")
        data = (
            tomllib.loads(text)
            if self.rubric.suffix.lower() == ".toml"
            else json.loads(text)
        )
        items = data.get("criteria", []) if isinstance(data, dict) else data
        criteria = [Criterion.model_validate(item) for item in items]
        if not criteria:
            raise ValueError(f"rubric file '{self.rubric}' lists no criteria")
        names = [criterion.name for criterion in criteria]
        if len(set(names)) != len(names):
            raise ValueError(
                f"rubric file '{self.rubric}' has duplicate criterion names"
            )
        if bad := [c.name for c in criteria if not 0 <= c.weight < math.inf]:
            raise ValueError(
                f"rubric '{self.rubric}' has negative or non-finite criterion "
                f"weights: {bad}"
            )
        if sum(criterion.weight for criterion in criteria) <= 0:
            raise ValueError(f"rubric '{self.rubric}' has no positive criterion weight")
        return criteria


class ScoreConfig(vf.BaseConfig):
    """How the judge's verdict composes with the taskset's own rewards on the
    solver's trace. Judge-only by default."""

    task_weight: float = 0.0
    """Scale applied to the taskset's own rewards; 1 keeps them next to the verdict."""
    judge_weight: float = 1.0
    """Weight of the judge's verdict in the solver's reward."""


class AgenticJudgeEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    """The solver agent. Its runtime must be a container:
    `--env.solver.runtime.type docker|prime`."""
    judge: vf.AgentConfig = vf.AgentConfig()
    """The judge agent. Under `isolated` it provisions its own box from this policy;
    under `shared` it plays in the solver's box and this policy is ignored."""
    topology: Literal["shared", "isolated"] = "shared"
    """Whether the judge grades in the solver's box or its own.

    `shared` places the judge in the box the agent worked in, so it can inspect that
    environment directly — at the cost of leaving every seam in it reachable. It is the
    default because it is what has been measured, and because a judge only gains from
    isolation once the task publishes what the judge needs.

    `isolated` boots a second box from the same image, carries the task's declared
    artifacts across, and grades there, so nothing the agent did to its environment can
    reach the grader. Opt in per taskset, and only once that taskset puts its evidence
    somewhere that travels: an artifact under `vf.CONVENTION_DIR` (see
    `capture_patch(write_path=...)`), a declared `TaskData.artifacts` path, or the trace
    record itself. A task whose judge hint says to run `git diff` in the box will find
    a pristine checkout here and grade every attempt as untouched."""
    task: JudgeTaskConfig = JudgeTaskConfig()
    score: ScoreConfig = ScoreConfig()


class AgenticJudgeEnv(vf.Env[AgenticJudgeEnvConfig]):
    def __init__(self, config: AgenticJudgeEnvConfig) -> None:
        # The judge inherits the solver's runtime policy unless it pins a container of
        # its own. Under `shared` that is definitional — its effective runtime IS the
        # solver's box, and aligning the config keeps the base env's subprocess warning
        # and the runtime stamped on its trace truthful. Under `isolated` it is the
        # fallback every other unpinned `AgentConfig` field already gets (harness,
        # model, sampling): `runtime` defaults to `SubprocessConfig` with no `None` to
        # distinguish unset from chosen, and a code-executing judge can never run on the
        # host anyway, so subprocess here always means "not specified".
        if config.topology == "shared" or isinstance(
            config.judge.runtime, vf.SubprocessConfig
        ):
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
                "agentic-judge plays a code-executing judge in its own sandbox, but "
                f"harness {judge.config.id!r} is a tool-less chat loop — a verdict "
                "that needs no execution is a plugged judge "
                "(--env.taskset.task.judges), not an agent."
            )
        if isinstance(self.config.solver.runtime, vf.SubprocessConfig):
            raise TypeError(
                "agentic-judge runs a code-executing solver in a container, but the "
                "solver resolves to the subprocess runtime; use "
                "--env.solver.runtime.type docker or prime"
            )

    async def setup(self, agents: vf.Agents) -> None:
        # The judge grades the policy; its tokens are never training data.
        agents.judge.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if self.config.topology == "shared":
            async with agents.solver.provision(task) as box:
                solution = await agents.solver.run(task, runtime=box)
                judge_task = JudgeTask.from_trace(solution, self.config.task, "shared")
                await agents.judge.run(judge_task, runtime=box)
            return

        # Collection is the barrier: once it returns, nothing downstream needs the
        # solver's box. Letting the context manager tear it down normally keeps an
        # episode at one box rather than two, which is what costs on a paid runtime.
        async with agents.solver.provision(task) as box:
            solution = await agents.solver.run(task, runtime=box)
            if not solution.ok:
                # The rollout errored, so its own scoring never ran either; grading it
                # would spend a second sandbox to reproduce a failure the trace already
                # records. Return rather than raise: `episode.ok` follows the failed
                # trace, and `episode_should_retry` classifies off that trace's own
                # error — an exception raised here would bury the real type.
                return
            collected = await vf.collect(box, task.data.artifacts)

        async with agents.judge.provision(task) as judge_box:
            await vf.restore(judge_box, collected)
            judge_task = JudgeTask.from_trace(solution, self.config.task, "isolated")
            await agents.judge.run(judge_task, runtime=judge_box)

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        by_agent = {t.agent_name: t for t in episode.traces}
        if "judge" not in by_agent:
            return  # solver errored; `run` skipped grading and the trace says why
        solution, verdict = by_agent["solver"], by_agent["judge"]
        data = verdict.info.get("verdict")
        if not isinstance(data, dict) or not isinstance(data.get("verdicts"), list):
            raise TypeError(
                f"no verdicts on the judge's trace (expected {VERDICT_FILE} with a "
                '"verdicts" list)'
            )
        criteria = self.config.task.criteria()
        by_criterion = {c.name: c for c in criteria}
        answers: dict[str, str] = {}
        for entry in data["verdicts"]:
            if not isinstance(entry, dict):
                raise TypeError(f"verdict entry {entry!r} is not an object")
            name = str(entry.get("name"))
            if name in answers:
                # Contradictory duplicates must not collapse to whichever came last.
                raise ValueError(f"judge answered criterion {name!r} more than once")
            answers[name] = str(entry.get("verdict"))
        if sorted(answers) != sorted(by_criterion):
            raise ValueError(
                f"judge verdicts name {sorted(answers)}, expected the rubric's "
                f"{sorted(by_criterion)}"
            )
        scores: dict[str, float] = {}
        for name, answer in answers.items():
            choices = by_criterion[name].choices
            # An off-menu answer is a judge failure, not a zero score.
            if answer not in choices:
                raise ValueError(
                    f"judge answered {answer!r} for '{name}', expected one of {choices}"
                )
            scores[name] = choices.index(answer) / (len(choices) - 1)
        for criterion in criteria:
            solution.record_metric(f"judge/{criterion.name}", scores[criterion.name])
        if self.config.score.task_weight != 1.0:
            for reward in solution.rewards.values():
                reward.weight *= self.config.score.task_weight
        total = sum(criterion.weight for criterion in criteria)
        reward = sum(c.weight * scores[c.name] for c in criteria) / total
        solution.record_reward("judge", reward, weight=self.config.score.judge_weight)
