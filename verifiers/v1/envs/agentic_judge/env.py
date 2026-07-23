"""agentic-judge: a solver plays the task, a judge verifies it in the same box.

A reusable env (`--env.id agentic-judge` over any taskset): the box is
provisioned from the solver's runtime policy, the solver plays the task in it,
and a code-executing judge then inspects the work as the agent left it, with
the solver's full trace record uploaded. The judge grades rubric criteria
(`[env.task]`: policy prompt, criteria file, trace path) and writes its
verdicts to `/tmp/verdict.json`; `finalize()` validates them strictly onto the
solver's trace — `judge/<name>` metrics plus a weighted-mean `judge` reward,
composed with the taskset's own rewards via `[env.score]` (judge-only by
default).
"""

import json
import math
import re
import tomllib
from pathlib import Path

from pydantic import field_validator

import verifiers.v1 as vf
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
    choices: list[str] = ["no", "yes"]
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


SANDBOX_NOTE = f"""\
## Your workspace

Your sandbox is the SAME box the graded agent worked in, in the state the agent
left it — its edits (and any scoring side effects) are applied. The agent's raw
trace record (JSON: messages, tool calls, and its `info` artifacts) is uploaded
at `{TRACE_FILE}`. The record is complete — it may also carry the task's own
scores/metrics and reference material (a gold answer, a reference solution,
held-out tests). Those are context, not your standard: recorded scores can be
wrong and references can be narrower than the task; do not over-index on how a
reference solves it. Your verdict is what YOU verified by execution."""


class JudgeTask(vf.Task):
    """The judge's verdict task: the solver task's world mirrored onto the minted
    row, the trace record written (and any stale verdict removed) before the
    judge starts, verdict scraped off the live box after it exits. `NEEDS_CONTAINER`
    keeps `Agent.run`'s per-task backstop aligned with the judge's declared need."""

    NEEDS_CONTAINER = True

    def __init__(self, data: vf.TaskData, files: dict[str, bytes]) -> None:
        super().__init__(data)
        self._files = files

    @classmethod
    def from_trace(
        cls, task: vf.Task, solution: vf.Trace, config: "AgenticJudgeEnvConfig"
    ) -> "JudgeTask":
        """Mint the judge's task from the solver's finished trace."""
        files = {TRACE_FILE: json.dumps(solution.to_record()).encode()}
        template = config.task.build_prompt()
        body = _render(template, prompt=task.data.prompt_text)
        if "{prompt}" not in template:
            # A policy that doesn't place the task statement itself still needs it.
            body += "\n\n" + _render(TASK_SECTION, prompt=task.data.prompt_text)
        prompt = "\n\n".join(
            [body, _verdict_section(config.task.criteria()), SANDBOX_NOTE]
        )
        return cls(
            vf.TaskData(
                idx=task.data.idx,
                prompt=prompt,
                image=task.data.image,
                workdir=task.data.workdir,
                resources=task.data.resources,
            ),
            files=files,
        )

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # A pre-seeded verdict (baked into the image, or left by the solver) must
        # never read as the judge's own; remove it before the judge starts.
        await runtime.run(["rm", "-f", VERDICT_FILE], env={})
        for path, content in self._files.items():
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
    rubric: Path | None = None
    """Criteria the judge grades against: a `.toml`/`.json` file with a
    `criteria` list — the plugged rubric judge's format, so the same rubric
    files work for both. None grades the single built-in `solved` criterion."""

    def build_prompt(self) -> str:
        if self.prompt is None:
            return GRADE_PROMPT + "\n\n" + TASK_SECTION
        path = Path(self.prompt)
        if isinstance(self.prompt, Path) or path.suffix in (".md", ".txt"):
            return path.read_text(encoding="utf-8")
        return str(self.prompt)

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
    """The solver agent. It owns the shared box, so its runtime must be a
    container: `--env.solver.runtime.type docker|prime`."""
    judge: vf.AgentConfig = vf.AgentConfig()
    """The judge agent. It plays in the solver's box; its own runtime policy is
    unused."""
    task: JudgeTaskConfig = JudgeTaskConfig()
    score: ScoreConfig = ScoreConfig()


class AgenticJudgeEnv(vf.Env[AgenticJudgeEnvConfig]):
    def __init__(self, config: AgenticJudgeEnvConfig) -> None:
        super().__init__(config)
        self._check_agents()
        # A missing policy file or a malformed rubric fails here, not mid-episode.
        config.task.build_prompt()
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
            raise ValueError(
                "agentic-judge plays its judge in the solver's box, but the solver "
                "(which provisions it) resolves to the subprocess runtime; use "
                "--env.solver.runtime.type docker or prime"
            )

    async def setup(self, agents: vf.Agents) -> None:
        # The judge grades the policy; its tokens are never training data.
        agents.judge.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        async with agents.solver.provision(task) as box:
            solution = await agents.solver.run(task, runtime=box)
            judge_task = JudgeTask.from_trace(task, solution, self.config)
            await agents.judge.run(judge_task, runtime=box)

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        by_agent = {t.agent_name: t for t in episode.traces}
        solution, verdict = by_agent["solver"], by_agent["judge"]
        data = verdict.info.get("verdict")
        if not isinstance(data, dict) or not isinstance(data.get("verdicts"), list):
            raise ValueError(
                f"no verdicts on the judge's trace (expected {VERDICT_FILE} with a "
                '"verdicts" list)'
            )
        criteria = self.config.task.criteria()
        by_criterion = {c.name: c for c in criteria}
        answers: dict[str, str] = {}
        for entry in data["verdicts"]:
            if not isinstance(entry, dict):
                raise ValueError(f"verdict entry {entry!r} is not an object")
            answers[str(entry.get("name"))] = str(entry.get("verdict"))
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
            for name in solution.rewards:
                solution.rewards[name] *= self.config.score.task_weight
        total = sum(criterion.weight for criterion in criteria)
        reward = sum(c.weight * scores[c.name] for c in criteria) / total
        solution.record_reward("judge", reward, weight=self.config.score.judge_weight)
