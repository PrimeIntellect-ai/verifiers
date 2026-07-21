"""agentic-judge: a solver plays the task, a code-executing judge verifies it in a sandbox.

Agent-as-judge as a reusable env (`--env.id agentic-judge` over any taskset). The
judge's task mirrors the solver task's world — same image/workdir/resources, in a
FRESH box in its original state (the solver's runtime is gone by judge time) —
with the graded transcript uploaded, so the judge reconstructs and tests the work
empirically, always in its own sandbox, never on the host.

The verdict channel is a file, not the chat: the judge writes
`{"score": 0-10, "reasoning": ...}` to `/tmp/verdict.json` in its box (a file
survives a chatty final reply), `JudgeTask.finalize` scrapes it off the live
runtime onto the judge's trace, and `score()` validates it strictly onto the
solver's trace — a missing, malformed, or off-scale verdict fails loudly instead
of clamping to full marks.
"""

import json
import math

import verifiers.v1 as vf
from verifiers.v1.harness import Harness

TRANSCRIPT_MD = "/tmp/transcript.md"
TRANSCRIPT_JSON = "/tmp/transcript.json"
VERDICT_FILE = "/tmp/verdict.json"

GRADE_PROMPT = f"""\
You are grading another agent's attempt at a task. Verify the work EMPIRICALLY:
reconstruct what the agent did from its transcript and test it with real
execution in your sandbox — never take the transcript's word for an outcome you
can check.

## The task the agent was given

{{prompt}}

## Your verdict

When you are done verifying, write your verdict as JSON to `{VERDICT_FILE}`:

    {{{{"score": <integer 0-10>, "reasoning": "<one paragraph>"}}}}

10 = the task is fully solved (you verified it); 0 = no progress. The score MUST
be an integer between 0 and 10 — nothing else is accepted."""


def _sandbox_note(solver: vf.TaskData) -> str:
    """What an agentic judge must know about its box before it starts verifying."""
    world = (
        f"a fresh instance of the same environment the graded agent worked in "
        f"(image {solver.image}), in its ORIGINAL state — the agent's edits are "
        "NOT applied; reconstruct them from the transcript to verify"
        if solver.image is not None
        else "your own — the graded agent worked elsewhere"
    )
    return (
        f"\n\n## Your workspace\nYour sandbox is {world}. The agent's full transcript "
        f"is uploaded at {TRANSCRIPT_MD} (rendered) and {TRANSCRIPT_JSON} (the raw "
        "trace record)."
    )


class JudgeTask(vf.Task):
    """The judge's verdict task: the solver task's world mirrored onto the minted
    row, transcript uploaded (and any stale verdict removed) before the judge
    starts, verdict scraped off the live box after it exits. `NEEDS_CONTAINER`
    keeps `Agent.run`'s per-task backstop aligned with the judge's declared need."""

    NEEDS_CONTAINER = True

    def __init__(self, data: vf.TaskData, files: dict[str, bytes]) -> None:
        super().__init__(data)
        self._files = files

    @classmethod
    def from_trace(cls, task: vf.Task, solution: vf.Trace) -> "JudgeTask":
        """Mint the judge's task from the solver's finished trace."""
        prompt = GRADE_PROMPT.format(prompt=task.data.prompt_text)
        return cls(
            vf.TaskData(
                idx=task.data.idx,
                prompt=prompt + _sandbox_note(task.data),
                image=task.data.image,
                workdir=task.data.workdir,
                resources=task.data.resources,
            ),
            files={
                TRANSCRIPT_MD: solution.transcript.encode(),
                TRANSCRIPT_JSON: json.dumps(solution.to_record()).encode(),
            },
        )

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
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


class AgenticJudgeEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    judge: vf.AgentConfig = vf.AgentConfig()
    """The judge agent. Its runtime must be a container:
    `--env.judge.harness.runtime.type docker|prime`."""
    weight: float = 1.0
    """Weight of the `judge` reward recorded on the solver's trace."""


class AgenticJudgeEnv(vf.Environment[AgenticJudgeEnvConfig]):
    def __init__(self, config: AgenticJudgeEnvConfig) -> None:
        super().__init__(config)
        self._check_judge_harness(self._harnesses["judge"])

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

    def brief(self, agents: vf.Agents) -> None:
        # The judge grades the policy; its tokens are never training data.
        agents.judge.trainable = False

    async def rollout(self, task: vf.Task, agents: vf.Agents) -> None:
        solution = await agents.solver.run(task)
        await agents.judge.run(JudgeTask.from_trace(task, solution))

    async def score(self, task: vf.Task, traces: list[vf.Trace]) -> None:
        """Record the scraped verdict on the SOLVER's trace. Strict on scale: an
        off-scale score raises (a judge answering `95` must not clamp to full
        marks), failing the env-rollout rather than scoring the solver wrong."""
        by_agent = {t.agent_name: t for t in traces}
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
        solution.record_reward("judge", float(score) / 10.0, self.config.weight)
