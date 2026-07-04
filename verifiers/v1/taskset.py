"""The taskset: produces typed tasks and owns scoring.

A `Taskset` is the data + judgement half of an environment. It yields typed
`Task`s, may expose tools via `tools`, and defines rewards/metrics as
decorated methods. All task framing lives in each task's user prompt (baked in by
`load_tasks`); the harness drives control flow.

It is the single judgement authority, scored at two granularities (execution lives in
the Rollout — per-rollout — and the Episode — group — which call these):
  - `score` runs `@reward`/`@metric` — plus any config-plugged judges (`config.judges`,
    see `verifiers.v1.judge`) — over one trace (in its live runtime).
  - `score_group` runs `@group_reward` over all the rollouts of one task at once —
    pairwise/preference rewards that compare samples.

For a heterogeneous taskset (different verification per task), have a single
`@reward` branch on a typed task field.
"""

import asyncio
from collections.abc import Mapping
from functools import cached_property
from typing import ClassVar, Generic, TypeVar

from pydantic import model_validator
from pydantic_config import BaseConfig

from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.errors import TasksetError, boundary
from verifiers.v1.judge import Judge, Judges
from verifiers.v1.types import EnvId, env_name
from verifiers.v1.runtimes import Runtime
from verifiers.v1.mcp import Toolset, User
from verifiers.v1.state import StateT
from verifiers.v1.task import TaskT
from verifiers.v1.trace import Trace


class TasksetConfig(BaseConfig):
    """Base taskset config. Subclass to add task-generation knobs."""

    id: EnvId = ""
    """The locally importable package selected by `--taskset.id`."""
    judges: Judges = []
    """Config-plugged judges, each resolved by `id` — a built-in (`reference`, `rubric`) or a
    locally importable package exporting a `Judge` subclass — and run by `score` after the
    taskset's own `@reward`s: grading plugged into any taskset/harness pair from the eval
    config alone, no taskset code. Each entry records its verdict in `trace.rewards` under
    its `name` with its `weight` (see `JudgeConfig`)."""

    @property
    def name(self) -> str:
        """The taskset's local package name."""
        return env_name(self.id)

    @model_validator(mode="before")
    @classmethod
    def _resolve_judges(cls, data):
        """Narrow each `judges` entry to the config type its `id` resolves to (mirrors
        `EnvConfig._resolve_plugins`), so judge-specific fields (e.g. rubric's `path`)
        validate against the real config instead of being rejected by the base type."""
        if not isinstance(data, dict) or not data.get("judges"):
            return data
        from verifiers.v1.loaders import judge_config_type

        entries = []
        for entry in data["judges"]:
            raw = entry.model_dump() if isinstance(entry, BaseConfig) else dict(entry)
            if not raw.get("id"):
                raise ValueError(
                    "each `judges` entry needs an `id` (a judge plugin: `reference`, "
                    "`rubric`, or a locally importable package)"
                )
            entries.append(judge_config_type(raw["id"]).model_validate(raw))
        data["judges"] = entries
        return data

    @model_validator(mode="after")
    def _check_judges(self) -> "TasksetConfig":
        """Validate the resolved `judges` — after the before-hook so class-level *defaults*
        (which never pass through it, e.g. a taskset config pre-plugging a judge) are held
        to the same rules: an `id` on every entry, and no two entries sharing a reward key
        (the second would clobber the first's verdict)."""
        names = []
        for entry in self.judges:
            if not entry.id:
                raise ValueError(
                    "each `judges` entry needs an `id` (a judge plugin: `reference`, "
                    "`rubric`, or a locally importable package)"
                )
            names.append(entry.name or env_name(entry.id))
        if duplicates := {name for name in names if names.count(name) > 1}:
            raise ValueError(
                f"`judges` entries share a reward key {sorted(duplicates)}; set a "
                "distinct `name` on each to keep both verdicts"
            )
        return self


ConfigT = TypeVar("ConfigT", bound=TasksetConfig)


class Taskset(Generic[TaskT, ConfigT, StateT]):
    """Generic over its task, config, and (optional) per-rollout `State` types, so `self.config`,
    `load_tasks`, and the trace's `state` are fully typed. `StateT` defaults to the base `State`, so a
    taskset that doesn't customize state writes just `Taskset[MyTask, MyConfig]`. Subclass: implement
    `load_tasks`, add @reward/@metric."""

    NEEDS_CONTAINER: ClassVar[bool] = False
    """Whether this taskset only runs in a container runtime (docker/prime). When True the
    Environment refuses the subprocess runtime — for tasksets whose work only makes sense
    inside a per-task image (e.g. a SWE repo sandbox)."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    def load_tasks(self) -> list[TaskT]:
        raise NotImplementedError

    @cached_property
    def judges(self) -> list[Judge]:
        """The plugged judges, built once from `config.judges` (each entry resolved by its
        `id` — see `JudgeConfig` / `verifiers.v1.judges`) and run by `score` after the
        decorated rewards."""
        from verifiers.v1.loaders import load_judge

        return [load_judge(entry) for entry in self.config.judges]

    def tools(self, task: TaskT) -> list[Toolset]:
        """Tool servers exposing this task's tools to the model — `vf.Toolset`s (classes with
        `@vf.tool` methods), each carrying its `config` (placement / runtime; a remote `url`
        for an already-running server). Empty by default; override to give a task tools."""
        return []

    def user(self, task: TaskT) -> User | None:
        """A user simulator for this task — structurally a tool server (an MCP server
        with a runtime), but driven by the framework, not exposed to the model. After
        each model turn the interception server calls its `respond` tool and injects the
        reply as a user turn. None by default; override to make a task a simulated
        multi-turn conversation (e.g. a TextArena game)."""
        return None

    async def setup(self, task: TaskT, trace: Trace, runtime: Runtime) -> None:
        """Prepare the live runtime for this task, after `runtime.start()` and before the
        harness runs. No-op by default; override to run per-task setup in the runtime (e.g.
        a SWE row checking out its base commit). Errors propagate and fail the rollout.

        Like the scoring hooks, `setup` declares the inputs it needs *by parameter name* and
        the framework injects them: any subset of `task`, `trace`, `runtime`. The trace (and
        its per-rollout `trace.state`) already exists when `setup` runs, so an override may
        stash per-rollout state there — e.g. `setup(self, task, trace, runtime)` or the
        legacy `setup(self, task, runtime)` both work."""
        return None

    async def finalize(self, task: TaskT, trace: Trace, runtime: Runtime) -> None:
        """Post-process the live runtime after the harness finishes, before scoring. No-op
        by default; override to do per-rollout work the rewards depend on — apply/commit the
        agent's diff, run a build, snapshot state, scrape runtime artifacts into `trace.info`.
        Runs while the runtime is still live (after generation, before `@reward`/`@metric`); the
        symmetric counterpart to `setup`. Errors propagate and fail the rollout."""
        return None

    async def validate(self, task: TaskT, runtime: Runtime) -> bool:
        """Check a task is well-formed and solvable, independent of any model rollout — run
        by the `validate` entrypoint, never during a rollout. Valid (True) by default;
        override to assert the ground truth holds (e.g. a SWE row applying its gold patch and
        running its tests, or gsm8k confirming the verifier accepts the gold answer). Runs in
        a live runtime started for the task with `setup` already applied (a pure-data check
        can ignore it). Return False — or raise — to mark the task invalid; the entrypoint
        records the reason (the raised error's message)."""
        return True

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        """Score one rollout: run all `@metric`, then `@reward`, then the config-plugged
        `judges` over its trace, concurrently within each phase. Each metric is recorded in
        `trace.metrics` (a number, or a mapping merged in); each reward and judge verdict
        (weighted — likewise a number or a mapping merged in) in `trace.rewards`, which
        `trace.reward` sums. Signals declare what they need — `task`, `trace`,
        `runtime` — so a reward is either a pure function of the trace or runs
        read/write/exec in that (still-live) runtime, e.g. a verifier script."""
        available = {"task": trace.task, "trace": trace, "runtime": runtime}
        async with boundary(TasksetError, f"taskset {type(self).__name__} scoring"):
            metrics = discover_decorated(self, "metric")
            metric_results = (
                [await invoke(fn, available) for fn in metrics]
                if len(metrics) < 2
                else await asyncio.gather(*(invoke(fn, available) for fn in metrics))
            )
            for fn, result in zip(metrics, metric_results):
                if isinstance(result, Mapping):
                    trace.record_metrics(result)
                else:
                    trace.record_metric(fn.__name__, result)
            rewards = discover_decorated(self, "reward")
            reward_results = (
                [await invoke(fn, available) for fn in rewards]
                if len(rewards) < 2
                else await asyncio.gather(*(invoke(fn, available) for fn in rewards))
            )
            for fn, result in zip(rewards, reward_results):
                weight = getattr(fn, "_vf_weight", 1.0)
                if isinstance(result, Mapping):
                    for name, value in result.items():
                        trace.record_reward(name, value, weight)
                else:
                    trace.record_reward(fn.__name__, result, weight)
            judges = self.judges
            judge_results = (
                [await invoke(judge.score, available) for judge in judges]
                if len(judges) < 2
                else await asyncio.gather(
                    *(invoke(judge.score, available) for judge in judges)
                )
            )
            for judge, result in zip(judges, judge_results):
                if isinstance(result, Mapping):
                    for name, value in result.items():
                        trace.record_reward(name, value, judge.config.weight)
                else:
                    trace.record_reward(judge.reward_name, result, judge.config.weight)

    async def score_group(self, traces: list[Trace]) -> None:
        """Score a group of rollouts of one task: run every `@group_reward` over all
        the traces at once (pairwise/preference rewards), each returning one score per
        trace, aligned to `traces`. A group reward declares what it needs — `task` (the
        shared task) and `traces` — and compares trace metadata (anything from the
        runtime is recorded per rollout as a `@metric` first). Scores are weighted into
        each trace's reward, alongside the per-rollout rewards. No-op without `@group_reward`s."""
        rewards = discover_decorated(self, "group_reward")
        if not rewards:
            return
        available = {"task": traces[0].task, "traces": traces}
        async with boundary(
            TasksetError, f"taskset {type(self).__name__} group scoring"
        ):
            reward_results = (
                [await invoke(fn, available) for fn in rewards]
                if len(rewards) < 2
                else await asyncio.gather(*(invoke(fn, available) for fn in rewards))
            )
            for fn, scores in zip(rewards, reward_results):
                weight = getattr(fn, "_vf_weight", 1.0)
                for trace, score in zip(traces, scores):
                    trace.record_reward(fn.__name__, score, weight)
