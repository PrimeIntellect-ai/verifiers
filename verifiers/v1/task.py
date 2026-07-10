"""Task data, configuration, behavior, and scoring.

Only `TaskData` travels on traces. `Task` wraps that data with runtime behavior and a
config; parameterize it as `Task[MyData, MyState, MyConfig]`.
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
    fn,
    name: str,
    result,
    section: str,
    weight: float | None = None,
) -> None:
    """Record a scalar or keyed scoring result and its runtime-only replay keys."""
    values = list(result.items()) if isinstance(result, Mapping) else [(name, result)]
    for key, value in values:
        if weight is None:
            trace.record_metric(key, value)
        else:
            trace.record_reward(key, value, weight)
    # Record runtime-only keys so offline replay can preserve them.
    if _requires_runtime(fn):
        offline = trace.info.setdefault("offline_keys", {})
        offline.setdefault(section, {}).setdefault(name, []).extend(
            key for key, _ in values
        )


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
    owner: str, config: BaseConfig, server_cls: type
) -> BaseConfig:
    cfg_cls = server_cls._config_cls()
    values = {name: getattr(config, name) for name in type(config).model_fields}
    # Prefer an exact field over a broader base-config match.
    matched = [name for name, value in values.items() if type(value) is cfg_cls]
    if not matched:
        matched = [name for name, value in values.items() if isinstance(value, cfg_cls)]
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
        return resolve_server_config(type(self).__name__, self.config, server_cls)

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
                _record_result(trace, fn, fn.__name__, result, "signals")
            reward_results = await invoke_all(rewards, available)
            for fn, result in zip(rewards, reward_results):
                _record_result(
                    trace,
                    fn,
                    fn.__name__,
                    result,
                    "signals",
                    getattr(fn, "_vf_weight", 1.0),
                )
            judge_results = await invoke_all(
                [judge.score for judge in judges], available
            )
            for judge, result in zip(judges, judge_results):
                _record_result(
                    trace,
                    judge.score,
                    judge.reward_name,
                    result,
                    "judges",
                    judge.config.weight,
                )

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
        truth), plus `@group_reward` keys (no group context offline — see `score_group`). A
        judge's entries restore only while that judge is still attached — a removed judge
        leaves no stale entry. This covers rewards and *returned* metric keys; metrics
        recorded by direct `record_metric` writes are unattributable and are restored by
        `replay` itself, wholesale, after re-scoring (fill-if-missing)."""
        recorded = trace.info.get("offline_keys")
        if not recorded:
            return
        # Only restore judge keys for judges still configured.
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
        for key in keys:
            if key in prior_rewards:
                trace.rewards[key] = prior_rewards[key]
            if key in prior_metrics:
                trace.metrics[key] = prior_metrics[key]

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
            # Replay has no group context, so preserve these scores by key.
            group_keys = [fn.__name__ for fn in rewards]
            for trace in traces:
                trace.info.setdefault("offline_keys", {})["group"] = group_keys


TaskT = TypeVar("TaskT", bound=Task)
