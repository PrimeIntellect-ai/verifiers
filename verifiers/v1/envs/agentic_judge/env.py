"""agentic-judge: a solver plays the task, a code-executing judge verifies it in a sandbox.

Agent-as-judge as a reusable env (`--env.id agentic-judge` over any taskset). By
default the two agents share one box (`--env.shared-runtime`, on): the box is
provisioned once from the solver's runtime policy, the solver plays the task in
it, and the judge then lands in the SAME box — the work exactly as the agent
left it (including any scoring side effects) — with the graded trace uploaded.
Note a borrowed box is never retried into, so per-agent rollout retries are off
in this mode (episode retries still apply).

With `--env.shared-runtime false`, the judge instead gets a FRESH box mirroring
the solver task's world (same image/workdir/resources), replayed to the solver's
STARTING state (the source task's setup runs; the solver's box is gone by judge
time), so the judge reconstructs the work from the uploaded trace — reproduce-first
verification in a pristine world, with no solver-controlled box state.

The grading policy is configurable (`--env.task.prompt`: inline text or a policy
file), but only the policy: the env always appends the verdict contract and the
workspace note, so a custom prompt cannot break verdict scraping. What lands in
the judge's box is the raw trace record (`--env.task.trace`, null to omit) —
it carries everything about the attempt (messages, tool calls, `trace.info`
artifacts such as a captured patch), and a policy that needs one of its fields
as a file instructs the judge to extract it. How the verdict composes with the
taskset's own rewards is `[env.score]` (judge-only by default).

The verdict is graded against rubric criteria — the same `criteria` file format
the plugged rubric judge reads (`{name, text, weight, choices}`, choices ordered
worst → best), one built-in `solved` criterion by default (`--env.task.rubric`
overrides). The channel is a file, not the chat: the judge writes
`{"verdicts": [{"name", "reason", "verdict"}, ...]}` to `/tmp/verdict.json` in
its box (a file survives a chatty final reply), `JudgeTask.finalize` scrapes it
off the live runtime onto the judge's trace, and the env's `finalize()`
validates it strictly onto the solver's trace — a missing verdict, an unknown
criterion, or an off-menu answer fails loudly instead of scoring the solver
wrong. Each criterion lands as a `judge/<name>` metric; the `judge` reward is
their weighted mean.
"""

import json
import math
import re
import tomllib
from pathlib import Path

from pydantic import field_validator

import verifiers.v1 as vf
from verifiers.v1.decorators import invoke
from verifiers.v1.types import StrictBaseModel

VERDICT_FILE = "/tmp/verdict.json"

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


def _sandbox_note(solver: vf.TaskData, trace_path: str | None, shared: bool) -> str:
    """What an agentic judge must know about its box before it starts verifying."""
    if shared:
        world = (
            "the SAME box the graded agent worked in, in the state the agent "
            "left it — its edits (and any scoring side effects) are applied"
        )
    else:
        world = (
            f"a fresh instance of the same environment the graded agent worked in "
            f"(image {solver.image}), in the same STARTING state the agent saw — the "
            "agent's edits are NOT applied; reconstruct them from the uploaded trace "
            "to verify"
            if solver.image is not None
            else "your own — the graded agent worked elsewhere"
        )
    uploaded = (
        f" The agent's raw trace record (JSON: messages, tool calls, and its "
        f"`info` artifacts) is uploaded at `{trace_path}`."
        if trace_path is not None
        else " Nothing about the graded attempt is uploaded; work from the prompt."
    )
    return f"## Your workspace\n\nYour sandbox is {world}.{uploaded}"


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
        source: vf.Task,
        shared: bool,
    ) -> None:
        super().__init__(data)
        self._files = files
        self._source = source
        self._shared = shared

    @classmethod
    def from_trace(
        cls, task: vf.Task, solution: vf.Trace, config: "AgenticJudgeEnvConfig"
    ) -> "JudgeTask":
        """Mint the judge's task from the solver's finished trace."""
        files: dict[str, bytes] = {}
        if config.task.trace is not None:
            record = solution.to_record()
            # The judge's verdict must be independent: never leak the graded
            # run's own scores (the judge anchors on them instead of verifying)
            # or the task row (it can carry ground truth — a gold answer, a
            # reference patch; the judge's prompt already states the task).
            record.pop("rewards", None)
            record.pop("metrics", None)
            record.pop("task", None)
            files[config.task.trace] = json.dumps(record).encode()

        template = config.task.grade_prompt()
        body = _render(template, prompt=task.data.prompt_text)
        if "{prompt}" not in template:
            # A policy that doesn't place the task statement itself still needs it.
            body += "\n\n" + _render(TASK_SECTION, prompt=task.data.prompt_text)
        shared = config.shared_runtime
        prompt = "\n\n".join(
            [
                body,
                _verdict_section(config.task.criteria()),
                _sandbox_note(task.data, config.task.trace, shared),
            ]
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
            source=task,
            shared=shared,
        )

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # On a fresh box the judge verifies against the state the solver STARTED
        # from, which is the image only after the source task's own setup (e.g. a
        # repo reset to the task's base commit) — replay it before seeding the
        # judge's files. On a shared box the solver's work IS the state; replaying
        # setup would destroy it.
        if not self._shared:
            await invoke(self._source.setup, {"trace": trace, "runtime": runtime})
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
    trace: str | None = "/tmp/trace.json"
    """Where in the judge's box to upload the solver's raw trace record; null to
    omit it. The record carries everything about the attempt (messages, tool
    calls, `trace.info` — e.g. a captured patch), so a policy that needs one of
    its fields as a file just instructs the judge to extract it."""

    def grade_prompt(self) -> str:
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
    judge: vf.AgentConfig = vf.AgentConfig()
    """The judge agent. With `shared_runtime` it plays in the solver's box (its
    own runtime policy is unused); otherwise its runtime must be a container:
    `--env.judge.harness.runtime.type docker|prime`."""
    task: JudgeTaskConfig = JudgeTaskConfig()
    score: ScoreConfig = ScoreConfig()
    shared_runtime: bool = True
    """Judge in the solver's live box (provisioned once from the solver's runtime
    policy, which must be a container): the judge inspects the work exactly as
    the agent left it. Set false to give the judge a fresh box mirroring the
    solver task's world, replayed to the solver's starting state — reconstruct-
    and-verify in a pristine world."""


class AgenticJudgeEnv(vf.Env[AgenticJudgeEnvConfig]):
    def __init__(self, config: AgenticJudgeEnvConfig) -> None:
        super().__init__(config)
        self._check_harnesses()
        # A missing policy file or a malformed rubric fails here, not mid-episode.
        config.task.grade_prompt()
        config.task.criteria()

    def _check_harnesses(self) -> None:
        """The judge executes real code, never on the host — refuse an impossible
        pairing at construction, not after burning a full solver run. The container
        requirement lands on whoever provisions the judge's box: the solver when
        the box is shared, the judge itself otherwise."""
        judge = self._harnesses["judge"]
        if not judge.EXECUTES_CODE:
            raise ValueError(
                "agentic-judge plays a code-executing judge in its own sandbox, but "
                f"harness {judge.config.id!r} is a tool-less chat loop — a verdict "
                "that needs no execution is a plugged judge "
                "(--env.taskset.task.judges), not an agent."
            )
        if self.config.shared_runtime and self.config.solver.replay is not None:
            raise ValueError(
                "a replayed solver leaves no box for the judge to share; use "
                "--env.shared-runtime false (the judge provisions its own box "
                "and verifies from the uploaded trace)"
            )
        box_owner = "solver" if self.config.shared_runtime else "judge"
        if isinstance(self._harnesses[box_owner].config.runtime, vf.SubprocessConfig):
            raise ValueError(
                f"agentic-judge plays its judge in a container, but the {box_owner} "
                "(which provisions the judge's box) resolves to the subprocess "
                f"runtime; use --env.{box_owner}.harness.runtime.type docker or "
                "prime"
                + (
                    ", or --env.shared-runtime false for a fresh judge box"
                    if self.config.shared_runtime
                    else ""
                )
            )

    async def setup(self, agents: vf.Agents) -> None:
        # The judge grades the policy; its tokens are never training data.
        agents.judge.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        if self.config.shared_runtime:
            async with agents.solver.provision(task) as box:
                solution = await agents.solver.run(task, runtime=box)
                judge_task = JudgeTask.from_trace(task, solution, self.config)
                await agents.judge.run(judge_task, runtime=box)
        else:
            solution = await agents.solver.run(task)
            await agents.judge.run(JudgeTask.from_trace(task, solution, self.config))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        """Grade the scraped verdict against the rubric and record it on the
        SOLVER's trace. Strict, mirroring the plugged rubric judge: exactly one
        verdict per criterion, matched by name, answered from that criterion's
        choices — anything else is a judge failure and fails the episode rather
        than scoring the solver wrong."""
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
