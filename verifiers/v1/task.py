"""Task data, configuration, behavior, and scoring.

`TaskData` is the wire half: a frozen pydantic model carrying everything a rollout's
row IS — the base fields plus your typed, task-specific fields. It rides on
`trace.task.data`, is what `traces.jsonl` stores, and what tool/user servers receive
over the `/task` channel. Subclass it per dataset.

`Task` is the behavior half: runtime prep (`setup`/`finalize`), server declarations
(`tools`/`user`), well-formedness (`validate`), and per-trace judgement
(`@reward`/`@metric` methods plus the plugged judges from `config.judges`, run by
`score`). Subclass per dataset and parameterize `Task[MyData, MyState, MyConfig]`
(all three default); judgement that compares sibling traces lives on
`Environment.score` instead.

A Task instance is shared across its rollouts (`-r n` runs hold the same instance),
so hooks must not stash per-rollout state on `self` — that lives on the trace
(`trace.state`).

On the wire only the data travels (plus the producing class's name,
`trace.task.type`): a saved row reads back as `WireTaskData` without importing the
taskset; a re-scoring consumer (`replay`) rebuilds the declared `TaskData` type and
wraps it in the declared `Task` — one task type per taskset.
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


def _reject_role_scoped(cls: type, why: str) -> None:
    """Refuse `@vf.reward(agent=...)`/`@vf.metric(agent=...)` on classes whose
    scoring never routes by agent (`Task`, `Harness`) — shared by their
    `__init_subclass__`."""
    for name, attr in vars(cls).items():
        if callable(attr) and getattr(attr, "_vf_agent", None) is not None:
            raise TypeError(
                f"{cls.__name__}.{name}: agent= belongs to an Environment's "
                f"cross-trace signals; {why} — drop agent="
            )


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
    """Judge plugins run after task rewards, set through `--env.taskset.task.judges`."""

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
    """The task's wire half: one row's pure data, a frozen pydantic model. Subclass
    per dataset to add typed task-specific fields next to the base fields; behavior
    lives on `Task`, which wraps this (`self.data`)."""

    model_config = ConfigDict(frozen=True)

    idx: int
    name: str | None = None
    description: str | None = None
    prompt: str | Messages | None = None
    """Initial user prompt; `None` lets the user simulator open the conversation. (A
    default, not just optional: the wire drops `None`s — `traces.jsonl` rows for
    prompt-less tasks must read back.)"""
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
    owner: str, config: BaseConfig, server_cls: type, *, sole: bool = True
) -> BaseConfig:
    """The config a declared server class is built with, resolved off `config`'s
    fields: exact type match, else — only for a `sole` declared server — the unique
    isinstance match, else a default-constructed one. Two matching fields raise; the
    `server_config` methods are the explicit-pairing override. The isinstance
    fallback is `sole`-gated because with several servers a subclass-typed field
    could silently pair with the wrong one."""
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

    def __init_subclass__(cls, **kwargs) -> None:
        """A task signal always scores its own trace, so `agent=` (an Environment
        concept) is refused at definition rather than silently ignored."""
        super().__init_subclass__(**kwargs)
        _reject_role_scoped(cls, "a Task signal always scores its own trace")

    def __init__(self, data: DataT, config: ConfigT | None = None) -> None:
        self.data = data
        self.config = config if config is not None else task_config_cls(type(self))()

    def plugged_judges(self) -> list[Judge]:
        from verifiers.v1.loaders import load_judge

        return [load_judge(config) for config in self.config.judges]

    def server_config(self, server_cls: type) -> BaseConfig:
        """The config a declared server class (`tools` / `user`) is built with (see
        `resolve_server_config`). Override to pair explicitly."""
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


TaskT = TypeVar("TaskT", bound=Task)
