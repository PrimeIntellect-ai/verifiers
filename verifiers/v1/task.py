"""Task data, configuration, behavior, and scoring.

`TaskData` is the wire half: a frozen pydantic model carrying everything a rollout's
row IS — the base fields (prompt, image, timeouts, judges) plus your typed,
task-specific fields (the reference answer, ground truths, ...). It is what rides on
`trace.task.data`, what `traces.jsonl` stores, and what tool/user servers receive over the
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
lives in the rollout engine — per-rollout — and the Episode — group — which call these):
  - `score` runs `@metric`/`@reward` — plus the plugged judges resolved from
    `config.judges` (see `verifiers.v1.judge`) — over one trace (in its live runtime).
  - `score_group` runs `@group_reward` over all the rollouts of this task at once —
    pairwise/preference rewards that compare samples.

A Task instance is shared across its rollouts (a group's `n` samples hold the same
instance), so hooks and scoring methods must not stash per-rollout state on `self` —
that lives on the trace (`trace.state`, typed via the `Task[..., MyState, ...]`
param).

On the wire only the data (plus the producing class's name, `trace.task.type`)
travels: a saved `trace.task.data` reads back as `WireTaskData`
(extra fields preserved) without importing the taskset; a consumer that re-scores
(e.g. `replay`) rebuilds the declared `TaskData` type and wraps it in the declared
`Task` — one task type per taskset.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Generic

from pydantic import ConfigDict, model_validator
from pydantic_config import BaseConfig
from typing_extensions import TypeVar

from verifiers.v1.decorators import discover_decorated, invoke_all
from verifiers.v1.errors import TaskError, boundary
from verifiers.v1.judge import Judges, check_judges, resolve_judges
from verifiers.v1.state import StateT
from verifiers.v1.types import Messages, StrictBaseModel, content_text
from verifiers.v1.utils.generic import generic_type

if TYPE_CHECKING:
    from verifiers.v1.judge import Judge
    from verifiers.v1.mcp import Toolset, User
    from verifiers.v1.runtimes import Runtime
    from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


def _requires_runtime(fn) -> bool:
    param = inspect.signature(fn).parameters.get("runtime")
    # A defaulted runtime parameter can still be called offline with None.
    return param is not None and param.default is inspect.Parameter.empty


def _record_result(
    trace: Trace,
    name: str,
    result,
    weight: float | None = None,
) -> None:
    """Record a scalar or keyed scoring result."""
    values = list(result.items()) if isinstance(result, Mapping) else [(name, result)]
    for key, value in values:
        if weight is None:
            trace.record_metric(key, value)
        else:
            trace.record_reward(key, value, weight)


class TaskResources(StrictBaseModel):
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
    """Optional per-task timeout overrides, in seconds."""

    model_config = ConfigDict(frozen=True)

    setup: float | None = None
    harness: float | None = None
    finalize: float | None = None
    scoring: float | None = None


class TaskConfig(BaseConfig):
    """Run-time knobs read by `Task` behavior.

    Subclass for server placement, judge, or scoring settings. Every field needs a
    default because constructing a task without a config builds the declared config type.
    Load-time dataset settings belong on `TasksetConfig` instead.
    """

    judges: Judges = []
    """Judge plugins run after task rewards, set through `--taskset.task.judges`."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_judges(cls, data):
        if isinstance(data, dict) and data.get("judges"):
            data["judges"] = resolve_judges(data["judges"])
        return data

    @model_validator(mode="after")
    def _check_judges(self) -> TaskConfig:
        check_judges(self.judges)
        return self


class TaskData(StrictBaseModel):
    """The task's wire half: one row's pure data, a frozen pydantic model. Subclass per
    dataset to add typed, task-specific fields (the reference answer, ground truths,
    per-row metadata) next to the base fields below. This is what `trace.task.data` holds,
    what `traces.jsonl` stores, and what tool/user servers receive over the `/task`
    channel — behavior lives on `Task`, which wraps this (`self.data`)."""

    model_config = ConfigDict(frozen=True)

    idx: int
    name: str | None = None
    description: str | None = None
    prompt: str | Messages | None
    """Initial user prompt; `None` lets the user simulator open the conversation."""
    system_prompt: str | None = None
    image: str | None = None
    workdir: str | None = None
    timeout: TaskTimeout = TaskTimeout()
    resources: TaskResources = TaskResources()
    sources: tuple[str, ...] = ()
    """Trace ids this task was derived from — set by agent programs that mint tasks out of
    earlier traces (a judge task from a solver trace, a solver task from a proposer trace).
    Empty for a dataset task."""
    relation: str | None = None
    """How this task relates to its `sources` (e.g. "judges", "solves", "monitors").
    Free-form; meaningful only when `sources` is non-empty."""

    @property
    def prompt_text(self) -> str:
        if isinstance(self.prompt, str):
            return self.prompt
        texts = [content_text(message.content) for message in self.prompt or []]
        return "\n\n".join(text for text in texts if text)


class WireTaskData(TaskData):
    """Wire form that preserves task-specific fields without importing the task class."""

    model_config = ConfigDict(extra="allow")


# No `default=`: an unparameterized `Trace`'s `task` field must serialize duck-typed
# (a defaulted TypeVar narrows pydantic's serialization to the base `TaskData`, silently
# dropping subclass fields from the wire).
DataT = TypeVar("DataT", bound=TaskData)
ConfigT = TypeVar("ConfigT", bound=TaskConfig, default=TaskConfig)


def task_data_cls(cls: type) -> type[TaskData]:
    """Resolve a task's `TaskData` specialization through its MRO, else `TaskData`."""
    return generic_type(cls, TaskData) or TaskData


def task_config_cls(cls: type) -> type[TaskConfig]:
    """Resolve a task's `TaskConfig` specialization through its MRO, else `TaskConfig`."""
    return generic_type(cls, TaskConfig) or TaskConfig


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
    return cfg_cls()


class Task(Generic[DataT, StateT, ConfigT]):
    """Behavior, lifecycle, servers, and scoring for one `TaskData` row.

    Parameterize as `Task[MyData, MyState, MyConfig]`. Construction accepts the row
    and an optional config; omitting config builds the declared config type. One task
    instance is shared across a rollout group, so per-rollout state belongs on the trace.
    """

    NEEDS_CONTAINER: ClassVar[bool] = False

    tools: ClassVar[tuple[type[Toolset], ...]] = ()

    user: ClassVar[type[User] | None] = None

    def __init__(self, data: DataT, config: ConfigT | None = None) -> None:
        self.data = data
        self.config = config if config is not None else task_config_cls(type(self))()

    def plugged_judges(self) -> list[Judge]:
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
        return [cls(self.server_config(cls)) for cls in type(self).tools]

    def user_server(self) -> User | None:
        cls = type(self).user
        return cls(self.server_config(cls)) if cls is not None else None

    async def setup(self, trace: Trace, runtime: Runtime) -> None:
        return None

    async def finalize(self, trace: Trace, runtime: Runtime) -> None:
        return None

    async def validate(self, runtime: Runtime) -> bool:
        return True

    async def score(
        self,
        trace: Trace,
        runtime: Runtime | None = None,
    ) -> None:
        judges = self.plugged_judges()
        available = {"task": self.data, "trace": trace}
        if runtime is not None:
            available["runtime"] = runtime

        async with boundary(TaskError, f"task {type(self).__name__} scoring"):
            metrics = discover_decorated(self, "metric")
            rewards = discover_decorated(self, "reward")
            if runtime is None:
                skipped = [
                    fn.__name__ for fn in (*metrics, *rewards) if _requires_runtime(fn)
                ] + [
                    judge.reward_name
                    for judge in judges
                    if _requires_runtime(judge.score)
                ]
                if skipped:
                    logger.info(
                        "score: no runtime — skipped runtime-dependent signals: %s",
                        skipped,
                    )
                metrics = [fn for fn in metrics if not _requires_runtime(fn)]
                rewards = [fn for fn in rewards if not _requires_runtime(fn)]
                judges = [
                    judge for judge in judges if not _requires_runtime(judge.score)
                ]

            metric_results = await invoke_all(metrics, available)
            for fn, result in zip(metrics, metric_results):
                _record_result(trace, fn.__name__, result)
            reward_results = await invoke_all(rewards, available)
            for fn, result in zip(rewards, reward_results):
                _record_result(
                    trace, fn.__name__, result, getattr(fn, "_vf_weight", 1.0)
                )
            judge_results = await invoke_all(
                [judge.score for judge in judges], available
            )
            for judge, result in zip(judges, judge_results):
                _record_result(trace, judge.reward_name, result, judge.config.weight)

    async def score_group(self, traces: list[Trace]) -> None:
        rewards = discover_decorated(self, "group_reward")
        if not rewards:
            return
        available = {"task": self.data, "traces": traces}
        async with boundary(TaskError, f"task {type(self).__name__} group scoring"):
            reward_results = await invoke_all(rewards, available)
            for fn, scores in zip(rewards, reward_results):
                if len(scores) != len(traces):
                    raise ValueError(
                        f"@group_reward {fn.__name__} returned {len(scores)} score(s) "
                        f"for {len(traces)} rollout(s); it must return one per trace"
                    )
                weight = getattr(fn, "_vf_weight", 1.0)
                for trace, score in zip(traces, scores):
                    trace.record_reward(fn.__name__, score, weight)


TaskT = TypeVar("TaskT", bound=Task)
