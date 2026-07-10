"""The task, split into data and behavior.

`TaskData` is the wire half: a frozen pydantic model carrying everything a rollout's
row IS — the base fields (prompt, image, timeouts, judges) plus your typed,
task-specific fields (the reference answer, ground truths, ...). It is what rides on
`trace.task`, what `traces.jsonl` stores, and what tool/user servers receive over the
`/task` channel. Subclass it per dataset.

`Task` is the behavior half: a plain class owning runtime prep (`setup`/`finalize`),
server declarations (`tools`/`user`), well-formedness (`validate`), and judgement
(`@reward`/`@metric`/`@group_reward` methods, run by `score`/`score_group`). It wraps
a `TaskData` plus a `TaskConfig`, both plain constructor arguments:

    task = MyTask(data)                        # config defaults to the declared type's
    task = MyTask(data, config=MyTaskConfig()) # or is injected, like Taskset/Harness/Judge
    task = MyTask.from_trace(trace)            # opt-in: derived from a finished rollout

Subclass per dataset and parameterize `Task[MyData, MyState, MyConfig]` (all three
default) — hooks and signals read the row off `self.data` and the knobs off
`self.config`. Because behavior lives on the task class, verification never branches
on a type field; a taskset yields one task type (its `load` constructs it), and
instances differ per row through their data.

The task is the single judgement authority, scored at two granularities (execution
lives in the Rollout — per-rollout — and the Episode — group — which call these):
  - `score` runs `@metric`/`@reward` — plus the plugged judges resolved from
    `config.judges` (see `verifiers.v1.judge`) — over one trace (in its live runtime).
  - `score_group` runs `@group_reward` over all the rollouts of this task at once —
    pairwise/preference rewards that compare samples.

A Task instance is shared across its rollouts (a group's `n` samples hold the same
instance), so hooks and scoring methods must not stash per-rollout state on `self` —
that lives on the trace (`trace.state`, typed via the `Task[..., MyState, ...]`
param).

On the wire only the data travels: a saved `trace.task` reads back as `WireTaskData`
(extra fields preserved) without importing the taskset; a consumer that re-scores
(e.g. `replay`) rebuilds the declared `TaskData` type and wraps it in the declared
`Task` — one task type per taskset.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Generic, Self, get_args

from pydantic import ConfigDict, model_validator
from pydantic_config import BaseConfig
from typing_extensions import TypeVar

from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.errors import TaskError, boundary
from verifiers.v1.judge import Judges, check_judges, resolve_judges
from verifiers.v1.state import StateT
from verifiers.v1.types import Messages, StrictBaseModel, content_text

if TYPE_CHECKING:
    from verifiers.v1.judge import Judge
    from verifiers.v1.mcp import Toolset, User
    from verifiers.v1.runtimes import Runtime
    from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


def _requires_runtime(fn) -> bool:
    """A signal with a default-less `runtime` parameter can't run without a live runtime.
    A `runtime` parameter with a default (e.g. `runtime=None`) is optional — `invoke`
    just omits it — so those still run offline."""
    param = inspect.signature(fn).parameters.get("runtime")
    return param is not None and param.default is inspect.Parameter.empty


class TaskResources(StrictBaseModel):
    """Runtime resources a task requests (all optional), in Modal's units. Applied to the
    runtime config where the field exists; a field the runtime doesn't support is warned
    about and ignored. Precedence: cli/toml > task > the runtime default (`None` here =
    use the runtime/provider default)."""

    model_config = ConfigDict(frozen=True)

    cpu: float | None = None
    """CPU cores."""
    memory: float | None = None
    """Memory in GB."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "A100:2" (type[:count])."""
    disk: float | None = None
    """Disk in GB (enforced by prime; advisory on docker/modal)."""


class TaskTimeout(StrictBaseModel):
    """Per-task wall-clock timeout overrides (seconds, all optional), one per rollout stage. Each
    merges with the eval's `timeout` (`TimeoutConfig`): cli/toml > this > default (no limit).
    Frozen, like `TaskResources`."""

    model_config = ConfigDict(frozen=True)

    setup: float | None = None
    """The task's `setup` hook."""
    harness: float | None = None
    """The harness run."""
    finalize: float | None = None
    """The task's `finalize` hook."""
    scoring: float | None = None
    """Verify + rewards/metrics."""


class TaskConfig(BaseConfig):
    """Rollout/scoring-time knobs the task reads off `self.config` — server placement
    (`ToolsetConfig`/`UserConfig` fields), judge endpoints, scoring parameters. Subclass to
    add your task's knobs; every field needs a default (a task constructed without one —
    e.g. via `Task.from_trace` — falls back to a default-constructed config). Load-time
    knobs (dataset, split, seed) belong on `TasksetConfig` instead — the task never needs
    them."""

    judges: Judges = []
    """Config-plugged judges, each resolved by `id` — a built-in (`reference`, `rubric`), a local
    package, or a hub `org/name[@version]` package exporting a `Judge` subclass: grading plugged
    into any taskset/harness pair from the eval config alone, no taskset code. `Task.score`
    resolves and runs them after the task's `@reward`s; each entry records its verdict in
    `trace.rewards` under its `name` with its `weight` (see `JudgeConfig`). Set via
    `--taskset.task.judges`."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_judges(cls, data):
        """Narrow each `judges` entry to the config type its `id` resolves to (see
        `judge.resolve_judges`), so judge-specific fields (e.g. rubric's `path`)
        validate against the real config instead of being rejected by the base type."""
        if isinstance(data, dict) and data.get("judges"):
            data["judges"] = resolve_judges(data["judges"])
        return data

    @model_validator(mode="after")
    def _check_judges(self) -> TaskConfig:
        """Validate the resolved `judges` — after the before-hook so class-level *defaults*
        (which never pass through it, e.g. a taskset config pre-plugging a judge) are held
        to the same rules (see `judge.check_judges`)."""
        check_judges(self.judges)
        return self


class TaskData(StrictBaseModel):
    """The task's wire half: one row's pure data, a frozen pydantic model. Subclass per
    dataset to add typed, task-specific fields (the reference answer, ground truths,
    per-row metadata) next to the base fields below. This is what `trace.task` holds,
    what `traces.jsonl` stores, and what tool/user servers receive over the `/task`
    channel — behavior lives on `Task`, which wraps this (`self.data`)."""

    model_config = ConfigDict(frozen=True)

    idx: int
    """Stable integer index of this example within its taskset."""
    name: str | None = None
    """Optional human-readable task name/label (for display/filtering)."""
    description: str | None = None
    """Optional human-readable task description."""
    prompt: str | Messages | None
    """The user message shown to the model (the task's question/framing). Usually a `str`; a
    `Messages` list seeds a full initial conversation (e.g. a user message carrying images) and
    is only accepted by harnesses that set `SUPPORTS_MESSAGE_PROMPT`. Required — set it
    explicitly to `None` to mean the task carries no prompt: the task's user simulator
    (`Task.user`) then opens the conversation, its first `respond` supplying the initial user
    turn before the model is ever called."""
    system_prompt: str | None = None
    """Optional system prompt. Harnesses that set `APPENDS_SYSTEM_PROMPT` emit it as a real
    system message (or their own mechanism); others prepend it to `prompt` (with a
    warning). See `Harness.resolve_prompt`."""
    image: str | None = None
    """Container image this task needs (e.g. its harbor environment). When set, the
    runtime must be a container (docker/prime): the Environment injects it into the
    runtime config and refuses the subprocess runtime, which has no container."""
    workdir: str | None = None
    """Working directory the harness and scoring run in — the Environment injects it into
    the runtime config's `workdir` (where the runtime supports one). For a containerized
    task whose image puts the working tree at a non-default path (e.g. a SWE row's
    `/workspace/<repo>`)."""
    timeout: TaskTimeout = TaskTimeout()
    """Optional per-task timeout overrides, one per rollout stage (merge with the eval's `timeout`)."""
    resources: TaskResources = TaskResources()
    """Optional runtime resources this task requests (applied where supported)."""

    @property
    def prompt_text(self) -> str:
        """The prompt as plain text — `prompt` itself when it's a `str`, the joined text
        content of a `Messages` prompt (images are dropped), `""` when the task carries no
        prompt. For consumers that need the task's framing as text regardless of the prompt
        form, e.g. the built-in judges' prompt templates."""
        if isinstance(self.prompt, str):
            return self.prompt
        texts = [content_text(message.content) for message in self.prompt or []]
        return "\n\n".join(text for text in texts if text)


class WireTaskData(TaskData):
    """A `TaskData` that accepts (and preserves) taskset-specific extra fields. Lets a
    `Trace` be typed on the wire — `Trace[WireTaskData]` — without importing the taskset,
    since the real `TaskData` subclass's extra fields land in `model_extra` instead of
    being rejected. A caller that re-scores rebuilds the real type via
    `task_data_cls(task_type(taskset_id))` first."""

    model_config = ConfigDict(extra="allow")


# No `default=`: an unparameterized `Trace`'s `task` field must serialize duck-typed
# (a defaulted TypeVar narrows pydantic's serialization to the base `TaskData`, silently
# dropping subclass fields from the wire).
DataT = TypeVar("DataT", bound=TaskData)
ConfigT = TypeVar("ConfigT", bound=TaskConfig, default=TaskConfig)


def _generic_arg(cls: type, base: type, default: type) -> type:
    """The `base` subclass a task class parameterizes — `Task[MyData, MyState, MyConfig]` —
    read off its generic bases, walking the MRO so a further subclass inherits it. Falls
    back to `default` when none is given. Mirrors `state_cls` (which reads the same
    generic for the `State` param)."""
    for klass in getattr(cls, "__mro__", [cls]):
        for orig in getattr(klass, "__orig_bases__", ()):
            for arg in get_args(orig):
                if isinstance(arg, type) and issubclass(arg, base):
                    return arg
    return default


def task_data_cls(cls: type) -> type[TaskData]:
    """The `TaskData` subclass a task class declares (`Task[MyData, ...]`); the base
    `TaskData` when none is given."""
    return _generic_arg(cls, TaskData, TaskData)


def task_config_cls(cls: type) -> type[TaskConfig]:
    """The `TaskConfig` subclass a task class declares (`Task[..., MyConfig]`); the base
    `TaskConfig` when none is given."""
    return _generic_arg(cls, TaskConfig, TaskConfig)


def resolve_server_config(
    owner: str, config: BaseConfig, server_cls: type, *, sole: bool = True
) -> BaseConfig:
    """The config a declared server class is built with, resolved off `config`'s fields:
    the field whose value is exactly the server's declared config type
    (`Toolset[MyConfig]` / `User[MyConfig]`), else the unique field whose value
    isinstance-matches it, else a default-constructed one. Two matching fields raise —
    the `server_config` methods (`Task` / `Taskset`) are the override points for explicit
    pairing. `owner` names the declaring class in errors. The isinstance fallback runs
    only when the owner declares a `sole` server class: with several, a subclass-typed
    field could silently pair with the WRONG server (e.g. a base-config server matching a
    sibling's narrowed field), so multi-server owners need exact type matches or a
    `server_config` override."""
    cfg_cls = server_cls._config_cls()
    values = {name: getattr(config, name) for name in type(config).model_fields}
    matched = [name for name, v in values.items() if type(v) is cfg_cls]
    if not matched and sole:
        matched = [name for name, v in values.items() if isinstance(v, cfg_cls)]
    if len(matched) > 1:
        raise TaskError(
            f"{owner}: ambiguous config for {server_cls.__name__} — config fields "
            f"{matched} all match {cfg_cls.__name__}; override `server_config` to pair "
            f"them explicitly"
        )
    if matched:
        return values[matched[0]]
    try:
        return cfg_cls()
    except Exception as exc:
        raise TaskError(
            f"{owner}: no {cfg_cls.__name__} to build {server_cls.__name__} with — add a "
            f"{cfg_cls.__name__} field to the config, or pass `config=...` at construction"
        ) from exc


class Task(Generic[DataT, StateT, ConfigT]):
    """The task's behavior half: hooks, server declarations, and `@reward`/`@metric`
    scoring over one row's `TaskData`. A plain class — subclass per dataset and
    parameterize `Task[MyData, MyState, MyConfig]` (all three default) so `self.data`,
    `trace.state`, and `self.config` are typed. Constructed from its two inputs like
    every other plugin (`Taskset(config)` / `Harness(config)` / `Judge(config)`):

        MyTask(data)                         # config = the declared type's defaults
        MyTask(data, config=MyTaskConfig())  # or injected
        MyTask.from_trace(trace)             # opt-in: derived from a finished rollout

    A taskset's `load()` constructs one per row with the eval's `TasksetConfig.task`.
    Hooks and signals read the row off `self.data` and the knobs off `self.config`; one
    instance is shared across a group's rollouts, so per-rollout state lives on
    `trace.state`, never on `self`."""

    NEEDS_CONTAINER: ClassVar[bool] = False
    """Whether this task only runs in a container runtime (docker/prime). When True the
    Environment refuses the subprocess runtime — for tasks whose work only makes sense
    inside a per-task image (e.g. a SWE repo sandbox)."""

    tools: ClassVar[tuple[type[Toolset], ...]] = ()
    """TASK-scoped tool server classes exposing this task's tools to the model —
    `vf.Toolset`s (classes with `@vf.tool` methods), each launched per rollout (its
    `ToolsetConfig`: colocated in the harness's runtime, or its own). Declarative: name
    the classes; the framework builds each instance with the config `server_config`
    resolves off `self.config`. Empty by default. An eval-wide server is declared on
    `Taskset.tools` instead — scope is where you register, not a flag."""

    user: ClassVar[type[User] | None] = None
    """The task's user simulator class — structurally a tool server (an MCP server
    with a runtime), but driven by the framework, not exposed to the model. After
    each model turn the interception server calls its `respond` tool and injects the
    reply as a user turn. Declarative like `tools`; None by default — set it to make
    the task a simulated multi-turn conversation (e.g. a TextArena game)."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Scope is the registration site; the config type must agree, or the mismatch
        # fails silently at runtime (a shared config's knobs ignored / a "shared" server
        # launched per rollout). Definition-time, so authors hit it on import.
        from verifiers.v1.mcp.toolset import SharedToolsetConfig

        for toolset in cls.tools:
            if issubclass(toolset._config_cls(), SharedToolsetConfig):
                raise TypeError(
                    f"{cls.__name__}.tools declares {toolset.__name__}, whose config is a "
                    "SharedToolsetConfig — an eval-level shared server belongs on "
                    "Taskset.tools; a task-scoped (per-rollout) server declares a ToolsetConfig"
                )

    def __init__(self, data: DataT, config: ConfigT | None = None) -> None:
        self.data = data
        if config is None:
            # Default to the *declared* config type's defaults, so a task constructed
            # anywhere is complete out of the box; a config that can't default fails
            # eagerly, at construction.
            try:
                config = task_config_cls(type(self))()  # type: ignore[assignment]
            except Exception as exc:
                raise TaskError(
                    f"{type(self).__name__} was built without a config and its "
                    f"{task_config_cls(type(self)).__name__} can't be default-"
                    f"constructed — pass `config=...`"
                ) from exc
        self.config = config

    @classmethod
    def from_trace(cls, trace: Trace, *, config: ConfigT | None = None) -> Self:
        """Derive a task from the trace of a previous rollout — the opt-in constructor
        for tasks that are not loaded from a taskset (e.g. a multi-agent step spawning a
        follow-up task from a finished trajectory). The trace is a *bare* `Trace`: its
        task/state need not be this task's declared types — the override decides what to
        read off it and how to build the new row. Not implemented by default; a task that
        implements it declares it can be spawned from a finished rollout. The config is
        the explicit `config` if given, else the declared type's defaults (the trace
        carries data only — configs come from whoever spawns the task)."""
        raise NotImplementedError(
            f"{cls.__name__} cannot be constructed from a trace — override "
            f"`from_trace` to define how a finished rollout spawns this task"
        )

    def plugged_judges(self) -> list[Judge]:
        """The runtime `Judge` objects for this task's plugged judges — resolved from
        `config.judges` alone (`--taskset.task.judges`; judges are config, never row
        data). Built fresh per call — a judge is a cheap value (its HTTP client is
        opened per call inside `Judge.complete` and closed when it returns), so nothing
        is shared or cached."""
        from verifiers.v1.loaders import load_judge

        return [load_judge(config) for config in self.config.judges]

    def server_config(self, server_cls: type) -> BaseConfig:
        """The config a declared server class (`tools` / `user`) is built with, resolved
        off `self.config` (see `resolve_server_config`: exact type match, else — for a
        sole declared server — unique isinstance match, else default-constructed).
        Override to pair explicitly (the escape hatch for exotic setups, e.g. two servers
        sharing one config type)."""
        declared = set(type(self).tools) | ({type(self).user} - {None})
        return resolve_server_config(
            type(self).__name__, self.config, server_cls, sole=len(declared) == 1
        )

    def tool_servers(self) -> list[Toolset]:
        """Build this task's tool servers: one instance per class in `tools`, each
        constructed with `server_config(cls)`. Called by the framework per rollout (and
        once per eval for `shared` placements); a Toolset instance is a launcher spec —
        the server itself runs as its own process (see `verifiers.v1.mcp`)."""
        return [cls(self.server_config(cls)) for cls in type(self).tools]

    def user_server(self) -> User | None:
        """Build this task's user simulator from the `user` declaration (None when the
        task doesn't declare one), constructed like a tool server."""
        cls = type(self).user
        return cls(self.server_config(cls)) if cls is not None else None

    async def setup(self, trace: Trace, runtime: Runtime) -> None:
        """Prepare the live runtime for this task, after `runtime.start()` and before the
        harness runs. No-op by default; override to run per-task setup in the runtime (e.g.
        a SWE row checking out its base commit — read the row off `self.data`). Errors
        propagate and fail the rollout.

        Like the scoring hooks, `setup` declares the inputs it needs *by parameter name* and
        the framework injects them: any subset of `trace`, `runtime`. The trace (and its
        per-rollout `trace.state`) already exists when `setup` runs, so an override may
        stash per-rollout state there — e.g. `setup(self, trace, runtime)` or just
        `setup(self, runtime)` both work."""
        return None

    async def finalize(self, trace: Trace, runtime: Runtime) -> None:
        """Post-process the live runtime after the harness finishes, before scoring. No-op
        by default; override to do per-rollout work the rewards depend on — apply/commit the
        agent's diff, run a build, snapshot state, scrape runtime artifacts into `trace.info`.
        Runs while the runtime is still live (after generation, before `@reward`/`@metric`); the
        symmetric counterpart to `setup`. Declares its inputs by parameter name, like `setup`.
        Errors propagate and fail the rollout."""
        return None

    async def validate(self, runtime: Runtime) -> bool:
        """Check the task is well-formed and solvable, independent of any model rollout — run
        by the `validate` entrypoint, never during a rollout. Valid (True) by default;
        override to assert the ground truth holds (e.g. a SWE row applying its gold patch and
        running its tests, or gsm8k confirming the verifier accepts the gold answer). Runs in
        a live runtime started for the task with `setup` already applied (a pure-data check
        can ignore it). Return False — or raise — to mark the task invalid; the entrypoint
        records the reason (the raised error's message)."""
        return True

    # ---- scoring ---------------------------------------------------------------

    async def score(
        self,
        trace: Trace,
        runtime: Runtime | None = None,
    ) -> None:
        """Score one rollout: run all `@metric`, then `@reward`, then the plugged judges
        (see `TaskConfig.judges`) over its trace, concurrently within each phase. Each
        metric is recorded in `trace.metrics` (a number, or a mapping merged in); each
        reward and judge verdict (weighted — likewise a number or a mapping merged in) in
        `trace.rewards`, which `trace.reward` sums. Signals declare what they need —
        `trace`, `runtime` (`self` is the task; the row is `self.data`) — so a reward is
        either a pure function of the trace or runs read/write/exec in that (still-live)
        runtime, e.g. a verifier script.

        `runtime` may be `None` — the offline path taken by `replay`, which re-scores a saved
        trace with no live runtime. Signals that declare a `runtime` parameter are then skipped
        (they can't run without one) and logged; trace-only `@metric`/`@reward`s and the plugged
        judges (which grade from the trace) still run, overwriting their own keys and leaving the
        skipped signals' previously-recorded values untouched. With a runtime present (the eval
        path) nothing is skipped."""
        judges = self.plugged_judges()
        # Judges receive the row (`task`): they read reference fields and `prompt_text`
        # off the data, never the behavior.
        available = {"task": self.data, "trace": trace}
        if runtime is not None:
            available["runtime"] = runtime

        def can_run(fn) -> bool:
            # A signal that *requires* a runtime can't run without one, so skip it when replaying.
            return runtime is not None or not _requires_runtime(fn)

        async with boundary(TaskError, f"task {type(self).__name__} scoring"):
            metrics = [fn for fn in discover_decorated(self, "metric") if can_run(fn)]
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
            rewards = [fn for fn in discover_decorated(self, "reward") if can_run(fn)]
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
            runnable_judges = [judge for judge in judges if can_run(judge.score)]
            if runtime is None:
                # Exactly the signals `can_run` filtered out above — those requiring a runtime
                # (a `runtime` param with no default). Signals whose `runtime` has a default run
                # offline and are not reported as skipped.
                skipped = self.offline_skipped()
                if skipped:
                    logger.info(
                        "score: no runtime — skipped runtime-dependent signals: %s",
                        skipped,
                    )
            judge_results = (
                [await invoke(judge.score, available) for judge in runnable_judges]
                if len(runnable_judges) < 2
                else await asyncio.gather(
                    *(invoke(judge.score, available) for judge in runnable_judges)
                )
            )
            for judge, result in zip(runnable_judges, judge_results):
                if isinstance(result, Mapping):
                    for name, value in result.items():
                        trace.record_reward(name, value, judge.config.weight)
                else:
                    trace.record_reward(judge.reward_name, result, judge.config.weight)
            if runtime is not None and isinstance(trace.info, dict):
                # Map each runtime-requiring signal to the reward/metric keys it actually
                # produced — a Mapping result records under its own keys, not the method
                # name — so an offline re-score (`replay`) can restore exactly these
                # (`offline_skipped` names the signals; this maps them to their keys).
                def keys(result, fallback: str) -> list[str]:
                    return list(result) if isinstance(result, Mapping) else [fallback]

                # Grouped (not a dict-comprehension union): same-named signals must
                # merge their keys, not overwrite each other's. Signals and judges are
                # recorded separately — on restore, a signal's keys are intrinsic to the
                # task, while a judge's only apply if that judge is still attached.
                # Only *returned* keys are attributable: signals run concurrently, so a
                # direct `record_metric` write can't be traced to its writer — `replay`
                # restores source metrics wholesale instead (fill-if-missing, after
                # re-scoring).
                signals: dict[str, list[str]] = {}
                for fn, result in (
                    *zip(metrics, metric_results),
                    *zip(rewards, reward_results),
                ):
                    if _requires_runtime(fn):
                        signals.setdefault(fn.__name__, []).extend(
                            keys(result, fn.__name__)
                        )
                judge_keys: dict[str, list[str]] = {}
                for judge, result in zip(runnable_judges, judge_results):
                    if _requires_runtime(judge.score):
                        judge_keys.setdefault(judge.reward_name, []).extend(
                            keys(result, judge.reward_name)
                        )
                if signals or judge_keys:
                    trace.info["offline_keys"] = {
                        "signals": signals,
                        "judges": judge_keys,
                    }

    def offline_skipped(self) -> list[str]:
        """The names of the signals `score` skips when re-scoring without a runtime (they
        declare a default-less `runtime` parameter): this task's `@metric`/`@reward`s plus
        its plugged judges."""
        signals = [
            fn.__name__
            for fn in (
                *discover_decorated(self, "metric"),
                *discover_decorated(self, "reward"),
            )
            if _requires_runtime(fn)
        ]
        judges = [
            judge.reward_name
            for judge in self.plugged_judges()
            if _requires_runtime(judge.score)
        ]
        return signals + judges

    def restore_offline(
        self,
        trace: Trace,
        prior_rewards: Mapping[str, float],
        prior_metrics: Mapping[str, float],
    ) -> None:
        """Pre-fill an offline re-score with the source run's runtime-only values: called
        by `replay` after clearing the prior scores and BEFORE `score`, so runtime-dependent
        entries survive — and trace-only signals that read them (e.g. a `@reward` reading a
        runtime `@metric`'s entry) see them while re-scoring. Restores exactly the keys the
        source run's runtime-requiring signals recorded (`info["offline_keys"]` — source-run
        truth, so it works even on a base-`Task` fallback that carries no behavior), plus
        `@group_reward` keys (no group context offline — see `score_group`); traces
        predating that map fall back to the signal names (`offline_skipped`). A judge's
        entries restore only while that judge is still attached — a removed judge leaves no
        stale entry. This covers rewards and *returned* metric keys; metrics recorded by
        direct `record_metric` writes are unattributable and are restored by `replay`
        itself, wholesale, after re-scoring (fill-if-missing)."""
        recorded = (
            trace.info.get("offline_keys") if isinstance(trace.info, dict) else None
        )
        if isinstance(recorded, Mapping):
            attached = {
                judge.reward_name
                for judge in self.plugged_judges()
                if _requires_runtime(judge.score)
            }
            keys = (
                [
                    key
                    for signal_keys in recorded.get("signals", {}).values()
                    for key in signal_keys
                ]
                + [
                    key
                    for name, judge_keys in recorded.get("judges", {}).items()
                    if name in attached
                    for key in judge_keys
                ]
                + list(recorded.get("group", []))
            )
        else:
            # Group rewards join the fallback too: like the runtime-requiring signals,
            # an offline re-score can't recompute them (they need the whole group).
            keys = self.offline_skipped() + [
                fn.__name__ for fn in discover_decorated(self, "group_reward")
            ]
        for key in keys:
            if key in prior_rewards and key not in trace.rewards:
                trace.rewards[key] = prior_rewards[key]
            if key in prior_metrics and key not in trace.metrics:
                trace.metrics[key] = prior_metrics[key]

    async def score_group(self, traces: list[Trace]) -> None:
        """Score a group of rollouts of this task: run every `@group_reward` over all
        the traces at once (pairwise/preference rewards), each returning one score per
        trace, aligned to `traces`. A group reward declares what it needs — `traces`
        (`self` is the shared task) — and compares trace metadata (anything from the
        runtime is recorded per rollout as a `@metric` first). Scores are weighted into
        each trace's reward, alongside the per-rollout rewards. No-op without `@group_reward`s."""
        rewards = discover_decorated(self, "group_reward")
        if not rewards:
            return
        available = {"task": self.data, "traces": traces}
        async with boundary(TaskError, f"task {type(self).__name__} group scoring"):
            reward_results = (
                [await invoke(fn, available) for fn in rewards]
                if len(rewards) < 2
                else await asyncio.gather(*(invoke(fn, available) for fn in rewards))
            )
            for fn, scores in zip(rewards, reward_results):
                if len(scores) != len(traces):
                    raise ValueError(
                        f"@group_reward {fn.__name__} returned {len(scores)} score(s) "
                        f"for {len(traces)} rollout(s); it must return one per trace"
                    )
                weight = getattr(fn, "_vf_weight", 1.0)
                for trace, score in zip(traces, scores):
                    trace.record_reward(fn.__name__, score, weight)
            # An offline re-score (`replay`) has no group context, so group rewards can
            # never be recomputed: record their keys for `restore_offline`. Merges into
            # the map `score` stamped (each rollout scores before the group does).
            group_keys = [fn.__name__ for fn in rewards]
            for trace in traces:
                if isinstance(trace.info, dict):
                    trace.info.setdefault("offline_keys", {})["group"] = group_keys


TaskT = TypeVar("TaskT", bound=Task)
