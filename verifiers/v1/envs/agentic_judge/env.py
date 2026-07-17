"""agentic-judge: a solver plays the task, a code-executing judge verifies it in a sandbox.

Agent-as-judge as a reusable env (`--env.id agentic-judge` over any taskset). The
judge's verdict task mirrors the solver task's world (same image/workdir/resources — a
FRESH box, in its original state; the solver's runtime is gone by judge time), with the
graded transcript uploaded (`/tmp/transcript.md` rendered, `/tmp/transcript.json` the
raw record) so the judge can reconstruct and test the work empirically — always in its
own sandbox, never on the host.

The verdict spec is a judge plugin (`--env.spec.id score|rubric|reference|org/name`,
the same registry as an `env.taskset.task.judges` entry): the prompt+parse a plugged
judge runs as one bare call is here executed as a real agent run, and `spec.verdict()`
lands on the SOLVER's trace exactly as the plugged tier records it. The spec's
`model`/`client`/`sampling` are ignored — the judge agent makes the calls; route it
with `--env.judge.*` (role-stamped, untrainable by default).

The judge seat is unpinned by default (the taskset's default harness) but must land in
a container — a subprocess-resolving judge is refused at construction (pin
`--env.judge.harness.runtime.type docker|prime`), as is a tool-less one (`null`): a
verdict that needs no execution is a plugged judge, not an agent.
"""

import json

from pydantic import SerializeAsAny, model_validator

import verifiers.v1 as vf
from verifiers.v1.env import _deep_merge
from verifiers.v1.harness import Harness
from verifiers.v1.judge import JudgeConfig
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
    """An agentic judge's verdict task: the solver task's world mirrored onto the
    minted row, with the graded transcript uploaded before the judge starts.
    `NEEDS_CONTAINER` keeps `Agent.run`'s per-task backstop aligned with the role's
    declared container need — judge-written code never runs on the host."""

    NEEDS_CONTAINER = True

    def __init__(self, data: vf.TaskData, files: dict[str, bytes]) -> None:
        super().__init__(data)
        self._files = files

    async def setup(self, trace, runtime):
        for path, content in self._files.items():
            await runtime.write(path, content)


class AgenticJudgeEnvConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    judge: vf.AgentConfig = vf.AgentConfig(trainable=False)
    """The judge seat. Unpinned, it runs the taskset's default harness (the same
    program the solver defaults to) in a mirror of the solver's world — its runtime
    must be a container: `--env.judge.harness.runtime.type docker|prime`."""
    spec: SerializeAsAny[JudgeConfig] = ScoreJudgeConfig(name="judge")
    """The verdict spec — a judge plugin's config, resolved by `--env.spec.id` exactly
    like an `env.taskset.task.judges` entry. Its `name`/`weight` set the reward key and
    weight on the solver's trace; a partial override (`--env.spec.view full_trace`)
    tunes the default spec, an explicit `--env.spec.id` swaps it."""

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

        self._spec = load_judge(self.config.spec)

    def _judge_harness(self) -> Harness:
        """The judge seat's resolved harness: its own pinned one, else the taskset's
        default."""
        from verifiers.v1.loaders import load_harness

        return load_harness(self._seat_harness(self.config.judge))

    def _check_judge_harness(self, harness: Harness) -> None:
        if not harness.EXECUTES_CODE:
            raise ValueError(
                "agentic-judge plays a code-executing judge in its own sandbox, but "
                f"harness {harness.config.id!r} is a tool-less chat loop — a verdict "
                "that needs no execution is a plugged judge "
                "(--env.taskset.task.judges), not an agent."
            )

    def roles(self):
        # The judge investigates with real execution — never on the host: its role
        # needs a container (fail-fast at construction; a judge resolving to the
        # subprocess runtime is refused, with the flag to set). The harness-kind
        # check runs first.
        self._check_judge_harness(self._judge_harness())
        return {
            "solver": vf.Role(self.config.solver),
            "judge": vf.Role(self.config.judge, mcp=False, container=True),
        }

    async def rollout(self, task, agents):
        solution = await agents["solver"].run(task)
        prompt = self._spec.render(task.data, solution)
        # The judge gets the solver task's world: a fresh box of the same image
        # (original state — the solver's edits live only in the solver's box), the
        # transcript uploaded, the prompt told where things stand.
        judge_task = JudgeTask(
            vf.TaskData(
                idx=task.data.idx,
                prompt=_noted(prompt, _sandbox_note(task.data)),
                image=task.data.image,
                workdir=task.data.workdir,
                resources=task.data.resources.model_copy(),
            ),
            files={
                TRANSCRIPT_MD: solution.transcript.encode(),
                TRANSCRIPT_JSON: json.dumps(solution.to_record()).encode(),
            },
        )
        verdict = await agents["judge"].run(judge_task)
        return {"solver": solution, "judge": verdict}

    async def score(self, task, views):
        """Parse the judge agent's reply through the spec and record it like the
        plugged tier would — a malformed verdict raises, failing the env-rollout
        (retryable) rather than scoring the solver 0."""
        solution, verdict = views["solver"], views["judge"]
        result = self._spec.verdict(task.data, solution, verdict.last_reply or "")
        _record_result(
            solution, self._spec.reward_name, result, self._spec.config.weight
        )
