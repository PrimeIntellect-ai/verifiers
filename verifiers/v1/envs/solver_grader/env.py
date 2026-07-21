"""solver-grader: a solver plays the task, a code-executing grader verifies it in a sandbox.

Agent-as-judge as a reusable env (`--env.id solver-grader` over any taskset). The
grader's verdict task mirrors the solver task's world — same image/workdir/resources,
in a FRESH box in its original state (the solver's runtime is gone by grading time) —
with the graded transcript uploaded, so the grader reconstructs and tests the work
empirically, always in its own sandbox, never on the host.

The verdict spec is a judge plugin (`--env.spec.id`, the same registry as an
`env.taskset.task.judges` entry): the prompt+parse a plugged judge runs as one bare
call is here executed as a real agent run, and `spec.verdict()` lands on the
SOLVER's trace exactly as the plugged tier records it.
"""

import json

from pydantic import SerializeAsAny, model_validator

import verifiers.v1 as vf
from verifiers.v1.judge import JudgeConfig
from verifiers.v1.judges.score import ScoreJudgeConfig
from verifiers.v1.task import _record_result
from verifiers.v1.utils.generic import deep_merge

TRANSCRIPT_MD = "/tmp/transcript.md"
TRANSCRIPT_JSON = "/tmp/transcript.json"


def _sandbox_note(solver: vf.TaskData) -> str:
    """What a grading agent must know about its box before it starts verifying."""
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


class GradeTask(vf.Task):
    """The grader's verdict task: the solver task's world mirrored onto the minted
    row, transcript uploaded before the grader starts. `NEEDS_CONTAINER` keeps
    `Agent.run`'s per-task validation aligned with the grader's actual need."""

    NEEDS_CONTAINER = True

    def __init__(self, data: vf.TaskData, files: dict[str, bytes]) -> None:
        super().__init__(data)
        self._files = files

    async def setup(self, trace, runtime):
        for path, content in self._files.items():
            await runtime.write(path, content)


class SolverGraderEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    grader: vf.AgentConfig = vf.AgentConfig()
    """The grader. Its runtime must be a container:
    `--env.grader.harness.runtime.type docker|prime`."""
    spec: SerializeAsAny[JudgeConfig] = ScoreJudgeConfig(name="grader")
    """The verdict spec — a judge plugin's config; its `name`/`weight` set the
    reward key and weight on the solver's trace. The spec's own model/client/
    sampling are ignored: the grader agent makes the calls (route via
    `--env.grader.*`)."""

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
                deep_merge(default.model_dump(exclude_none=True), raw)
            )
        return data


class SolverGraderEnv(vf.Environment[SolverGraderEnvConfig]):
    def __init__(self, config: SolverGraderEnvConfig) -> None:
        super().__init__(config)
        from verifiers.v1.loaders import load_judge

        # The spec drives the grader's task; a misconfigured spec or grader
        # surfaces at runtime, on its own run (no upfront compilation).
        self._spec = load_judge(self.config.spec)

    def setup(self, agents):
        # The grader grades the policy; its tokens are never training data.
        agents.grader.trainable = False

    async def run(self, task, agents):
        solution = await agents.solver.run(task)
        prompt = self._spec.render(task.data, solution)
        # A fresh box of the solver task's image, original state (the solver's
        # edits live only in its own box), transcript uploaded.
        grade_task = GradeTask(
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
        await agents.grader.run(grade_task)

    async def score(self, task, traces):
        """Parse the grader's reply through the spec and record it like the plugged
        tier would — a malformed verdict raises, failing the episode (retryable)
        rather than scoring the solver 0."""
        by_agent = {t.agent_name: t for t in traces}
        solution, verdict = by_agent["solver"], by_agent["grader"]
        result = self._spec.verdict(task.data, solution, verdict.last_reply)
        _record_result(
            solution, self._spec.reward_name, result, self._spec.config.weight
        )
