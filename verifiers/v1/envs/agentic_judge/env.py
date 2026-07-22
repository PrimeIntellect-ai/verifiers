"""agentic-judge: a solver plays the task, a code-executing judge verifies it in a sandbox.

Agent-as-judge as a reusable env (`--env.id agentic-judge` over any taskset). The
judge's task mirrors the solver task's world — same image/workdir/resources, in a
FRESH box replayed to the solver's STARTING state (the source task's setup runs,
then the solver's runtime is gone by judge time) — with the graded transcript
uploaded, so the judge reconstructs and tests the work empirically, always in its
own sandbox, never on the host.

The grading policy is configurable (`--env.prompt` / `--env.prompt_file`), but only
the policy: the env always appends the verdict contract and the workspace note, so
a custom prompt cannot break verdict scraping. What lands in the judge's box is
configurable too (`[env.uploads]`) — the rendered transcript, the raw trace record,
and any `trace.info` values (e.g. a captured patch) as real files.

The verdict channel is a file, not the chat: the judge writes
`{"score": 0-10, "reasoning": ...}` to `/tmp/verdict.json` in its box (a file
survives a chatty final reply), `JudgeTask.finalize` scrapes it off the live
runtime onto the judge's trace, and the env's `finalize()` validates it strictly
onto the solver's trace — a missing, malformed, or off-scale verdict fails loudly instead
of clamping to full marks.
"""

import json
import logging
import math
import re
from pathlib import Path

from pydantic import model_validator

import verifiers.v1 as vf
from verifiers.v1.decorators import invoke
from verifiers.v1.harness import Harness

logger = logging.getLogger(__name__)

VERDICT_FILE = "/tmp/verdict.json"

GRADE_PROMPT = """\
You are grading another agent's attempt at a task. Verify the work EMPIRICALLY:
reconstruct what the agent did from its transcript and test it with real
execution in your sandbox — never take the transcript's word for an outcome you
can check."""

TASK_SECTION = """\
## The task the agent was given

{prompt}"""

VERDICT_SECTION = f"""\
## Your verdict

When you are done verifying, write your verdict as JSON to `{VERDICT_FILE}`:

    {{"score": <integer 0-10>, "reasoning": "<one paragraph>"}}

10 = the task is fully solved (you verified it); 0 = no progress. The score MUST
be an integer between 0 and 10 — nothing else is accepted."""


def _render(template: str, **fields: str) -> str:
    """Substitute documented placeholders in one pass over the original template —
    str.format would crash on any literal brace in a custom prompt, and sequential
    replaces would re-scan substituted values. An unknown placeholder stays as written."""
    pattern = re.compile(r"\{(" + "|".join(map(re.escape, fields)) + r")\}")
    return pattern.sub(lambda m: fields[m.group(1)], template)


def _sandbox_note(solver: vf.TaskData, files: dict[str, str]) -> str:
    """What an agentic judge must know about its box before it starts verifying."""
    world = (
        f"a fresh instance of the same environment the graded agent worked in "
        f"(image {solver.image}), in the same STARTING state the agent saw — the "
        "agent's edits are NOT applied; reconstruct them from the uploaded files "
        "to verify"
        if solver.image is not None
        else "your own — the graded agent worked elsewhere"
    )
    listing = "".join(f"\n- `{path}` — {what}" for path, what in files.items())
    uploaded = (
        f" These files about the graded attempt are uploaded:{listing}"
        if files
        else " Nothing about the graded attempt is uploaded; work from the prompt."
    )
    return f"## Your workspace\n\nYour sandbox is {world}.{uploaded}"


class JudgeTask(vf.Task):
    """The judge's verdict task: the solver task's world mirrored onto the minted
    row, the configured uploads written (and any stale verdict removed) before the
    judge starts, verdict scraped off the live box after it exits. `NEEDS_CONTAINER`
    keeps `Agent.run`'s per-task backstop aligned with the judge's declared need."""

    NEEDS_CONTAINER = True

    def __init__(
        self, data: vf.TaskData, files: dict[str, bytes], source: vf.Task
    ) -> None:
        super().__init__(data)
        self._files = files
        self._source = source

    @classmethod
    def from_trace(
        cls, task: vf.Task, solution: vf.Trace, config: "AgenticJudgeEnvConfig"
    ) -> "JudgeTask":
        """Mint the judge's task from the solver's finished trace."""
        uploads = config.uploads
        files: dict[str, bytes] = {}
        described: dict[str, str] = {}
        if uploads.transcript is not None:
            files[uploads.transcript] = solution.transcript.encode()
            described[uploads.transcript] = "the agent's transcript (rendered)"
        if uploads.trace is not None:
            record = solution.to_record()
            # The judge's verdict must be independent: never leak the graded
            # run's own scores (the judge anchors on them instead of verifying)
            # or the task row (it can carry ground truth — a gold answer, a
            # reference patch; the judge's prompt already states the task).
            record.pop("rewards", None)
            record.pop("metrics", None)
            record.pop("task", None)
            files[uploads.trace] = json.dumps(record).encode()
            described[uploads.trace] = "the raw trace record (JSON)"
        for key, path in uploads.info.items():
            value = solution.info.get(key)
            if value is None:
                logger.warning(
                    "upload skipped: trace.info[%r] is absent on the solver trace", key
                )
                continue
            content = value if isinstance(value, str) else json.dumps(value)
            files[path] = content.encode()
            described[path] = f"`{key}` from the graded agent's trace"

        template = config.grade_prompt()
        body = _render(template, prompt=task.data.prompt_text)
        if "{prompt}" not in template:
            # A policy that doesn't place the task statement itself still needs it.
            body += "\n\n" + _render(TASK_SECTION, prompt=task.data.prompt_text)
        prompt = "\n\n".join(
            [body, VERDICT_SECTION, _sandbox_note(task.data, described)]
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
        )

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # The judge verifies against the state the solver STARTED from, which is
        # the image only after the source task's own setup (e.g. a repo reset to
        # the task's base commit) — replay it before seeding the judge's files.
        await invoke(self._source.setup, {"trace": trace, "runtime": runtime})
        # A pre-seeded verdict (baked into the image) must never read as the
        # judge's own; remove it before the judge starts.
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
                'be writing {"score": <0-10>, "reasoning": ...} there'
            ) from e
        trace.info["verdict"] = json.loads(raw)


class UploadsConfig(vf.BaseConfig):
    """What from the solver's finished attempt lands in the judge's box, as files."""

    transcript: str | None = "/tmp/transcript.md"
    """Where to upload the rendered transcript; null to omit it."""
    trace: str | None = "/tmp/transcript.json"
    """Where to upload the raw trace record; null to omit it."""
    info: dict[str, str] = {}
    """`trace.info` keys to upload, mapped to sandbox paths — e.g.
    `patch = "/tmp/solution.patch"` hands the judge a captured git diff as a
    real file. An absent key is skipped with a warning, not an error."""


class AgenticJudgeEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    judge: vf.AgentConfig = vf.AgentConfig()
    """The judge agent. Its runtime must be a container:
    `--env.judge.harness.runtime.type docker|prime`."""
    prompt: str | None = None
    """Grading-policy override. Replaces only the policy body — the verdict
    contract and workspace note are always appended, so a custom policy cannot
    break verdict scraping. May reference `{prompt}` (the solver task's prompt);
    if it doesn't, the task statement is appended after the policy."""
    prompt_file: Path | None = None
    """Grading-policy file override, mutually exclusive with `prompt`."""
    uploads: UploadsConfig = UploadsConfig()
    judge_weight: float = 1.0
    """Weight of the judge's verdict in the solver's reward."""
    task_reward_weight: float = 1.0
    """Scale applied to the taskset's own rewards on the solver's trace; 0 makes
    the judge's verdict the only reward."""

    @model_validator(mode="after")
    def check_prompt_source(self) -> "AgenticJudgeEnvConfig":
        if self.prompt is not None and self.prompt_file is not None:
            raise ValueError("set `prompt` or `prompt_file`, not both")
        return self

    def grade_prompt(self) -> str:
        if self.prompt is not None:
            return self.prompt
        if self.prompt_file is not None:
            return self.prompt_file.read_text(encoding="utf-8")
        return GRADE_PROMPT + "\n\n" + TASK_SECTION


class AgenticJudgeEnv(vf.Env[AgenticJudgeEnvConfig]):
    def __init__(self, config: AgenticJudgeEnvConfig) -> None:
        super().__init__(config)
        self._check_judge_harness(self._harnesses["judge"])
        config.grade_prompt()  # a missing prompt_file fails here, not mid-episode

    def _check_judge_harness(self, harness: Harness) -> None:
        """The judge executes real code, never on the host — refuse an impossible
        judge at construction, not after burning a full solver run."""
        if not harness.EXECUTES_CODE:
            raise ValueError(
                "agentic-judge plays a code-executing judge in its own sandbox, but "
                f"harness {harness.config.id!r} is a tool-less chat loop — a verdict "
                "that needs no execution is a plugged judge "
                "(--env.taskset.task.judges), not an agent."
            )
        if isinstance(harness.config.runtime, vf.SubprocessConfig):
            raise ValueError(
                "agentic-judge plays its judge in a container (JudgeTask mirrors "
                "the solver task's image), but the judge resolves to the "
                "subprocess runtime; use --env.judge.harness.runtime.type docker "
                "or prime."
            )

    async def setup(self, agents: vf.Agents) -> None:
        # The judge grades the policy; its tokens are never training data.
        agents.judge.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        solution = await agents.solver.run(task)
        await agents.judge.run(JudgeTask.from_trace(task, solution, self.config))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        """Record the scraped verdict on the SOLVER's trace. Strict on scale: an
        off-scale score raises (a judge answering `95` must not clamp to full
        marks), failing the episode rather than scoring the solver wrong."""
        by_agent = {t.agent_name: t for t in episode.traces}
        solution, verdict = by_agent["solver"], by_agent["judge"]
        data = verdict.info.get("verdict")
        if not isinstance(data, dict):
            raise ValueError(
                f"no verdict on the judge's trace (expected {VERDICT_FILE})"
            )
        score = data.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or math.isnan(float(score))
            or not 0 <= float(score) <= 10
        ):
            raise ValueError(
                f"verdict score {score!r} is not on the 0-10 scale; refusing to "
                "clamp or coerce it"
            )
        if self.config.task_reward_weight != 1.0:
            for name in solution.rewards:
                solution.rewards[name] *= self.config.task_reward_weight
        solution.record_reward(
            "judge", float(score) / 10.0, weight=self.config.judge_weight
        )
