"""judge: a solver plays the task, a judge agent grades the finished attempt.

The agent-as-judge wrapper (`--env.id judge` over any taskset). The verdict spec is a
JUDGE PLUGIN (`--env.spec.id score|rubric|reference|org/name` — the same registry as a
`taskset.task.judges` entry): the prompt+parse a plugged judge runs as one bare call
is here executed as a real agent run. `spec.render()` becomes the judge role's task,
the agent speaks (one `direct`-harness call by default; point `--env.judge.harness.id`
at a real harness and the judge investigates with tools), and `spec.verdict()` parses
its final reply — recorded on the SOLVER's trace exactly as the plugged tier records
it (reward key + weight from the spec's config; a rubric spec also lands its
per-criterion metrics). Write your grading criteria once, choose the execution mode
per run.

Default spec: the built-in `score` judge (one 0-10 verdict) under the reward key
`judge`, reading the solver's final reply; a rubric spec defaults to the whole
transcript (`--env.spec.view full_trace`), which is how an agentic solver's *process*
gets judged. The judge's own trace is role-stamped and untrainable by default.

A judge that executes code (any harness beyond the tool-less chat loops,
`Harness.EXECUTES_CODE`) is always played in its own sandbox, never on the host: its
role needs a container — a judge harness left on the default subprocess runtime rides
the run's own container runtime, and a fully-subprocess run must pin one
(`--env.judge.harness.runtime.type docker`). Its verdict task mirrors the solver
task's world (same image/workdir/resources — a FRESH box, in its original state), and
the graded transcript is uploaded into it (`/tmp/transcript.md` rendered,
`/tmp/transcript.json` the raw trace record) so the judge can reconstruct and test the
agent's work empirically. The solver's runtime is still gone by judge time — a judge
that must inspect the solver's live box is a custom env (`provision()` + borrowed
`runtime=`), not this wrapper.

Unlike the plugged tiers (`config.judges` — a bare call inside the rollout;
`vf.Judge` — a call in your own reward code), the verdict here is a real, inspectable,
budgeted agent trace. The spec's `model`/`client`/`sampling` are ignored — the judge
AGENT makes the calls; route it with `--env.judge.*`.
"""

import json
from functools import cached_property

from pydantic import SerializeAsAny, model_validator

import verifiers.v1 as vf
from verifiers.v1.env import _deep_merge
from verifiers.v1.judge import JudgeConfig
from verifiers.v1.judges.score import ScoreJudgeConfig
from verifiers.v1.task import _record_result

TRANSCRIPT_MD = "/tmp/transcript.md"
TRANSCRIPT_JSON = "/tmp/transcript.json"


class JudgeParams(vf.EnvParams):
    solver: vf.AgentConfig = vf.AgentConfig()
    judge: vf.AgentConfig = vf.AgentConfig(
        harness=vf.HarnessConfig(id="direct"), trainable=False
    )
    spec: SerializeAsAny[JudgeConfig] = ScoreJudgeConfig(name="judge")
    """The verdict spec — a judge plugin's config, resolved by `--env.spec.id` exactly
    like a `taskset.task.judges` entry. Its `name`/`weight` set the reward key and
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


class JudgeEnv(vf.Environment[JudgeParams]):
    def __init__(self, config: vf.EnvConfig) -> None:
        super().__init__(config)
        from verifiers.v1.loaders import load_judge

        self._spec = load_judge(self.params.spec)

    @cached_property
    def _agentic(self) -> bool:
        """Whether the judge investigates with real execution — a harness that hands
        the model local execution (`EXECUTES_CODE`) — which is what forces the
        sandbox rules; a tool-less judge stays a bare model actor."""
        from verifiers.v1.loaders import load_harness

        cfg = self.params.judge.harness
        harness = (
            self.harness
            if cfg is None or cfg == self.config.harness
            else load_harness(cfg)
        )
        return harness.EXECUTES_CODE

    def roles(self):
        # The topology: the solver plays the dataset (the taskset's needs apply);
        # the judge plays env-minted verdict tasks. A tool-less judge (the default
        # `direct`) is a bare model actor — it pairs with any taskset, tools and
        # containers included. A judge that executes code is never played on the
        # host: its role needs a container (fail-fast at construction), and a judge
        # harness left on the default subprocess runtime rides the run's own
        # container runtime — `--env.judge.harness.id default` on a docker run puts
        # the judge in docker with no further flags.
        judge = self.params.judge
        if (
            self._agentic
            and judge.harness is not None
            and isinstance(judge.harness.runtime, vf.SubprocessConfig)
            and not isinstance(self.config.harness.runtime, vf.SubprocessConfig)
        ):
            judge = judge.model_copy(
                update={
                    "harness": judge.harness.model_copy(
                        update={"runtime": self.config.harness.runtime}
                    )
                }
            )
        return {
            "solver": vf.Role(self.params.solver),
            "judge": vf.Role(judge, mcp=False, container=self._agentic),
        }

    async def rollout(self, task, agents):
        solution = await agents["solver"].run(task)
        prompt = self._spec.render(task.data, solution)
        if self._agentic:
            # A code-running judge gets the solver task's world: a fresh box of the
            # same image (original state — the solver's edits live only in the
            # solver's box), the transcript uploaded, the prompt told where things
            # stand.
            judge_task = JudgeTask(
                vf.TaskData(
                    idx=task.data.idx,
                    prompt=_noted(prompt, _sandbox_note(task.data)),
                    image=task.data.image,
                    workdir=task.data.workdir,
                    resources=task.data.resources.model_copy(),
                    sources=(solution.id,),
                    relation="judges",
                ),
                files={
                    TRANSCRIPT_MD: solution.transcript.encode(),
                    TRANSCRIPT_JSON: json.dumps(solution.to_record()).encode(),
                },
            )
        else:
            judge_task = vf.Task(
                vf.TaskData(
                    idx=task.data.idx,
                    prompt=prompt,
                    sources=(solution.id,),
                    relation="judges",
                )
            )
        verdict = await agents["judge"].run(judge_task)
        return [solution, verdict]

    async def score(self, task, traces):
        """Parse the judge agent's reply through the spec and record it like the
        plugged tier would — a malformed verdict raises, failing the env-rollout
        (retryable) rather than scoring the solver 0."""
        solution, verdict = traces
        result = self._spec.verdict(task.data, solution, verdict.last_reply or "")
        _record_result(
            solution, self._spec.reward_name, result, self._spec.config.weight
        )
