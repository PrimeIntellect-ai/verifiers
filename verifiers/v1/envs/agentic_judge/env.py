"""agentic-judge: a solver plays the task, a code-executing judge verifies it in a sandbox.

Agent-as-judge as a reusable env (`--env.id agentic-judge` over any taskset). The
judge's verdict task mirrors the solver task's world — same image/workdir/resources,
in a FRESH box in its original state (the solver's runtime is gone by judge time) —
with the graded transcript uploaded, so the judge reconstructs and tests the work
empirically, always in its own sandbox, never on the host.

The verdict spec is a judge plugin (`--env.spec.id`, the same registry as an
`env.taskset.task.judges` entry): the prompt+parse a plugged judge runs as one bare
call is here executed as a real agent run, and `spec.verdict()` lands on the
SOLVER's trace exactly as the plugged tier records it.
"""

import json

from pydantic import SerializeAsAny, model_validator

import verifiers.v1 as vf
from verifiers.v1.env import _deep_merge
from verifiers.v1.harness import Harness
from verifiers.v1.judge import Judge, JudgeConfig
from verifiers.v1.judges.score import ScoreJudgeConfig
from verifiers.v1.task import _record_result

TRANSCRIPT_MD = "/tmp/transcript.md"
TRANSCRIPT_JSON = "/tmp/transcript.json"


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


def _noted(prompt: str | vf.Messages, note: str) -> str | vf.Messages:
    if isinstance(prompt, str):
        return prompt + note
    return [*prompt, vf.UserMessage(content=note.strip())]


class JudgeTask(vf.Task):
    """The judge's verdict task: the solver task's world mirrored onto the minted
    row, transcript uploaded before the judge starts. `NEEDS_CONTAINER` keeps
    `Agent.run`'s per-task backstop aligned with the role's declared need."""

    NEEDS_CONTAINER = True

    def __init__(self, data: vf.TaskData, files: dict[str, bytes]) -> None:
        super().__init__(data)
        self._files = files

    async def setup(self, trace, runtime):
        for path, content in self._files.items():
            await runtime.write(path, content)


class AgenticJudgeEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    judge: vf.AgentConfig = vf.AgentConfig()
    """The judge seat. Its runtime must be a container:
    `--env.judge.harness.runtime.type docker|prime`."""
    spec: SerializeAsAny[JudgeConfig] = ScoreJudgeConfig(name="judge")
    """The verdict spec — a judge plugin's config; its `name`/`weight` set the
    reward key and weight on the solver's trace. The spec's own model/client/
    sampling are ignored: the judge agent makes the calls (route via
    `--env.judge.*`)."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_spec(cls, data):
        if not isinstance(data, dict) or not isinstance(data.get("spec"), dict):
            return data
        from verifiers.v1.loaders import judge_config_type

        raw = data["spec"]
        if raw.get("id"):  # an explicit id swaps the spec (narrowing to its config)
            data["spec"] = judge_config_type(raw["id"]).model_validate(raw)
        else:  # a partial override tunes the default spec, never resets it
            default = cls.model_fields["spec"].default
            data["spec"] = type(default).model_validate(
                _deep_merge(default.model_dump(exclude_none=True), raw)
            )
        return data


class AgenticJudgeEnv(vf.Environment[AgenticJudgeEnvConfig]):
    def __init__(self, config: AgenticJudgeEnvConfig) -> None:
        super().__init__(config)
        from verifiers.v1.loaders import load_judge

        self._check_judge_harness(self._harnesses["judge"])
        self._spec = load_judge(self.config.spec)
        # The spec drives the judge agent's task: refuse a render-less one at
        # construction, not after burning a full solver run.
        if type(self._spec).render is Judge.render:
            raise ValueError(
                f"agentic-judge runs the spec's `render` prompt as its judge "
                f"agent's task, but judge {self.config.spec.id!r} implements no "
                "`render` — a score-only judge belongs on the plugged tier "
                "(--env.taskset.task.judges), not this env."
            )

    def _check_judge_harness(self, harness: Harness) -> None:
        """The judge executes real code, never on the host — refuse an impossible
        judge seat at construction, not after burning a full solver run."""
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
                "the solver task's image), but the judge seat resolves to the "
                "subprocess runtime; use --env.judge.harness.runtime.type docker "
                "or prime."
            )

    def brief(self, agents):
        # The judge grades the policy; its tokens are never training data.
        agents["judge"].trainable = False

    async def rollout(self, task, agents):
        solution = await agents["solver"].run(task)
        prompt = self._spec.render(task.data, solution)
        # A fresh box of the solver task's image, original state (the solver's
        # edits live only in its own box), transcript uploaded.
        judge_task = JudgeTask(
            vf.TaskData(
                idx=task.data.idx,
                prompt=_noted(prompt, _sandbox_note(task.data)),
                image=task.data.image,
                workdir=task.data.workdir,
                resources=task.data.resources,
            ),
            files={
                TRANSCRIPT_MD: solution.transcript.encode(),
                TRANSCRIPT_JSON: json.dumps(solution.to_record()).encode(),
            },
        )
        verdict = await agents["judge"].run(judge_task)
        return {"solver": solution, "judge": verdict}

    async def score(self, task, views):
        """Parse the judge's reply through the spec and record it like the plugged
        tier would — a malformed verdict raises, failing the env-rollout
        (retryable) rather than scoring the solver 0."""
        solution, verdict = views["solver"], views["judge"]
        result = self._spec.verdict(task.data, solution, verdict.last_reply)
        _record_result(
            solution, self._spec.reward_name, result, self._spec.config.weight
        )
