"""judge: a solver plays the task, a bare judge agent grades the finished attempt.

The agent-as-judge wrapper (`--env.id judge` over any taskset). The verdict spec is a
JUDGE PLUGIN (`--env.spec.id score|rubric|reference|org/name` — the same registry as a
`taskset.task.judges` entry): the prompt+parse a plugged judge runs as one bare call
is here executed as a real agent run — one in-process `direct`-harness call by
default — and `spec.verdict()` parses its final reply, recorded on the SOLVER's trace
exactly as the plugged tier records it (reward key + weight from the spec's config; a
rubric spec also lands its per-criterion metrics). Write your grading criteria once,
choose the execution mode per run.

Default spec: the built-in `score` judge (one 0-10 verdict) under the reward key
`judge`, reading the solver's final reply (`--env.spec.view full_trace` shows it the
whole transcript instead). The judge's own trace is role-stamped and untrainable by
default.

The judge here is a bare model actor — a tool-less chat loop, so it pairs with any
taskset, tools and containers included. A judge that EXECUTES CODE is a different
env: `--env.id agentic-judge` plays it in its own sandbox with the graded transcript
uploaded; this env refuses a code-executing judge harness at construction.

Unlike the plugged tiers (`config.judges` — a bare call inside the rollout;
`vf.Judge` — a call in your own reward code), the verdict here is a real, inspectable,
budgeted agent trace. The spec's `model`/`client`/`sampling` are ignored — the judge
AGENT makes the calls; route it with `--env.judge.*`.
"""

from pydantic import SerializeAsAny, model_validator

import verifiers.v1 as vf
from verifiers.v1.env import _deep_merge
from verifiers.v1.harness import Harness
from verifiers.v1.judge import JudgeConfig
from verifiers.v1.judges.score import ScoreJudgeConfig
from verifiers.v1.task import _record_result


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


class JudgeEnv(vf.Environment[JudgeParams]):
    def __init__(self, config: vf.EnvConfig) -> None:
        super().__init__(config)
        from verifiers.v1.loaders import load_judge

        self._spec = load_judge(self.params.spec)

    def _judge_harness(self) -> Harness:
        """The judge seat's resolved harness: its own pinned one, else the taskset's
        default — never the run-level `--harness.*` (a multi-agent env refuses it)."""
        from verifiers.v1.loaders import load_harness

        return load_harness(self._seat_harness(self.params.judge))

    def _check_judge_harness(self, harness: Harness) -> None:
        if harness.EXECUTES_CODE:
            raise ValueError(
                "the judge env's judge is a bare model actor, but harness "
                f"{harness.config.id!r} executes code — judge-written code never "
                "runs on the host. Use --env.id agentic-judge: it plays the judge "
                "in its own sandbox with the graded transcript uploaded."
            )

    def roles(self):
        # The topology: the solver plays the dataset (the taskset's needs apply);
        # the judge grades env-minted verdict tasks as a bare model actor — it
        # pairs with any taskset, tools and containers included. The harness-kind
        # check runs here so it fires before the role-needs validation.
        self._check_judge_harness(self._judge_harness())
        return {
            "solver": vf.Role(self.params.solver),
            "judge": vf.Role(self.params.judge, mcp=False, container=False),
        }

    async def rollout(self, task, agents):
        solution = await agents["solver"].run(task)
        verdict_task = vf.Task(
            vf.TaskData(
                idx=task.data.idx,
                prompt=self._spec.render(task.data, solution),
                sources=(solution.id,),
                relation="judges",
            )
        )
        verdict = await agents["judge"].run(verdict_task)
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
