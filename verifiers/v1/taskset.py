"""The taskset: a thin loader that yields typed tasks.

A `Taskset` is the data half of an environment: config in, tasks out. `load()` is
the main hook that builds each task:

    def load(self) -> Iterable[MyTask]:
        for row in ...:
            yield MyTask(MyData(prompt=..., ...), self.config.task)

`load` may also be a generator for infinite tasksets. There is a one-to-one
mapping between taskset and task type, i.e. a taskset may only yield one task
type. Iterating the taskset sets each task's `idx` to its position in `load()`.

Lazy views pick which tasks an iteration yields, chainable in any order:

    taskset.include(idx=["0:100"]).exclude(names=["broken"]).shuffle(seed=0).take(5)

`select(SelectConfig)` chains them in the config's fixed order; the eval, debug,
validate and GEPA entrypoints and prime-rl all select through it.
"""

from __future__ import annotations

import copy
import itertools
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Generic, Self

from typing_extensions import TypeVar

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.configs.select import SelectConfig, TaskMatchConfig
from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.task import Task, TaskT
from verifiers.v1.utils.generic import concrete_type

if TYPE_CHECKING:
    from verifiers.v1.mcp import Toolset

logger = logging.getLogger(__name__)

TasksetConfigT = TypeVar("TasksetConfigT", bound=TasksetConfig, default=TasksetConfig)

Transform = Callable[[Iterator[Any]], Iterator[Any]]


class Taskset(ABC, Generic[TaskT, TasksetConfigT]):
    INFINITE: bool = False
    """Whether `load()` yields tasks forever. A view can still bound the iteration
    (see `bounded`)."""

    transform: Transform | None = None
    """Iteration transform carried by the views (see `_view`)."""
    _bounded: bool | None = None
    _ordered: bool = True
    """Whether the view still yields tasks in `load()` order (no shuffle yet)."""

    def __init__(self, config: TasksetConfigT) -> None:
        self.config = config
        override = config.system_prompt
        self.system_prompt = override.read_text() if override is not None else None
        declared = (
            type(config).model_fields["network"].get_default(call_default_factory=True)
        )
        self.network_default: NetworkPolicyConfig | None = declared
        # Only a value that differs from the config class's own default can have come
        # from TOML/CLI: a run's config crosses full-dump boundaries that lose
        # `model_fields_set`, so set-ness cannot tell the two apart.
        self.network_override: NetworkPolicyConfig | None = (
            config.network if config.network != declared else None
        )

    @abstractmethod
    def load(self) -> Iterable[TaskT]:
        """Build and yield the taskset's tasks; may be a generator (see module doc)."""

    @property
    def bounded(self) -> bool:
        """Whether iterating ends: `load()` is finite, or a view caps it (`take`, or
        `include` with only closed `idx` ranges)."""
        return not self.INFINITE if self._bounded is None else self._bounded

    def __iter__(self) -> Iterator[TaskT]:
        """Lazily iterate `load()` with each task's `idx` set to its position, the
        config-layer system prompt and the resolved network policy applied (a TOML/CLI
        `network` replaces, else the task's own, else the taskset's declared default),
        then the views' transform. The views see the final task data, so a `keys` match
        compares the keys that traces record. This is the read path; `load` is the
        subclass hook."""
        update = (
            {} if self.system_prompt is None else {"system_prompt": self.system_prompt}
        )
        override, default = self.network_override, self.network_default

        def policy(task: TaskT) -> NetworkPolicyConfig | None:
            if override is not None:
                return override
            return task.data.network if task.data.network is not None else default

        tasks: Iterator[TaskT] = (
            task.with_data(idx=idx, network=policy(task), **update)
            for idx, task in enumerate(self.load())
        )
        return tasks if self.transform is None else self.transform(tasks)

    def include(self, **match: Any) -> Self:
        """A view keeping only the tasks `match` names by `idx`, `ids`, `keys` or
        `names` (see `TaskMatchConfig`). An identity match on an unbounded view could
        read forever waiting for a match, so it raises there."""
        config = TaskMatchConfig(**match)
        if not self.bounded and (config.ids or config.keys or config.names):
            raise ValueError(
                f"{type(self).__name__} is infinite - include by ids, keys or names "
                "may never end; bound it first with closed include idx ranges"
            )
        # Positions arrive in increasing order until a shuffle, so reading can stop
        # past the last closed idx range.
        stop = config.idx_stop if self._ordered else None
        view = self._view(lambda tasks: _match(tasks, config, True, "include", stop))
        if config.idx_stop is not None:
            view._bounded = True
        return view

    def exclude(self, **match: Any) -> Self:
        """A view dropping the tasks `match` names (see `include`). Dropping every
        position from some start on leaves an unbounded view nothing to yield past
        it, so that raises there."""
        config = TaskMatchConfig(**match)
        if not self.bounded and config.idx_tail:
            raise ValueError(
                f"{type(self).__name__} is infinite - excluding an open idx range "
                "drops every task after its start; exclude a closed range instead"
            )
        return self._view(lambda tasks: _match(tasks, config, False, "exclude", None))

    def shuffle(self, seed: int = 0) -> Self:
        """A shuffled view under `seed` (materializes on iteration); raises on an
        unbounded view — bound it first with `take` or closed `include` idx ranges.
        `select` applies `limit` after the shuffle, so there only closed
        `include.idx` ranges can bound it."""
        if not self.bounded:
            raise ValueError(
                f"{type(self).__name__} is infinite - cannot shuffle; bound it first "
                "with closed include idx ranges (e.g. --select.include.idx 0:1000), "
                "or with take(n) before shuffle() in Python"
            )

        def shuffled(tasks: Iterator[TaskT]) -> Iterator[TaskT]:
            materialized = list(tasks)
            random.Random(seed).shuffle(materialized)
            return iter(materialized)

        view = self._view(shuffled)
        view._ordered = False
        return view

    def skip(self, num_tasks: int) -> Self:
        """A view without the first `num_tasks` tasks."""
        return self._view(lambda tasks: itertools.islice(tasks, num_tasks, None))

    def take(self, num_tasks: int) -> Self:
        """A lazy, always-finite view of the first `num_tasks` tasks."""
        view = self._view(lambda tasks: itertools.islice(tasks, num_tasks))
        view._bounded = True
        return view

    def select(self, config: SelectConfig) -> Self:
        """The views `config` describes, chained in its fixed order: `include`,
        `exclude`, `shuffle`, `skip`, `take`."""
        view = self
        if not config.include.empty:
            view = view.include(**config.include.model_dump())
        if not config.exclude.empty:
            view = view.exclude(**config.exclude.model_dump())
        if config.shuffle:
            view = view.shuffle(config.seed)
        if config.skip:
            view = view.skip(config.skip)
        if config.limit is not None:
            view = view.take(config.limit)
        return view

    def _view(self, transform: Transform) -> Self:
        """A shallow copy of this taskset iterating through `transform`, composed
        onto any transform this taskset already carries."""
        clone = copy.copy(self)
        prev = self.transform
        clone.transform = (
            transform if prev is None else lambda tasks: transform(prev(tasks))
        )
        return clone

    @classmethod
    def task_type(cls) -> type[Task]:
        return concrete_type(cls, Task, origin=Taskset) or Task

    @classmethod
    def toolsets(cls, config: TasksetConfigT) -> list[Toolset]:
        """Tool servers shared by all tasks in the taskset (one global instance
        per server, reused across an environment worker's rollouts), each
        constructed with its config off `config` — override and wire explicitly:

            @classmethod
            def toolsets(cls, config: MyConfig) -> list[vf.Toolset]:
                return [SearchToolset(config.tools)]
        """
        return []


def _match(
    tasks: Iterator[TaskT],
    match: TaskMatchConfig,
    keep: bool,
    label: str,
    stop: int | None,
) -> Iterator[TaskT]:
    """The tasks `match` names (`keep`) or does not name, reading no further than
    position `stop`. Once reading ends, warns about entries that matched no task, as
    these are usually typos."""
    ranges = match.idx_ranges()
    wanted = {"ids": set(match.ids), "keys": set(match.keys), "names": set(match.names)}
    found: dict[str, set[str]] = {field: set() for field in wanted}
    hit_ranges: set[int] = set()
    for task in tasks:
        idx = task.data.idx
        if stop is not None and idx >= stop:
            break
        in_ranges = {i for i, r in enumerate(ranges) if idx in r}
        hit_ranges |= in_ranges
        values = {
            "ids": task.data.id,
            "keys": task.key if match.keys else None,
            "names": task.data.name,
        }
        hits = [f for f, value in values.items() if value in wanted[f]]
        for field in hits:
            found[field].add(values[field])
        if bool(hits or in_ranges) == keep:
            yield task
        if stop is not None and idx + 1 >= stop:
            break
    missing: dict[str, list] = {
        f: sorted(wanted[f] - found[f]) for f in wanted if wanted[f] - found[f]
    }
    if unmatched := [match.idx[i] for i in range(len(ranges)) if i not in hit_ranges]:
        missing["idx"] = unmatched
    if missing:
        logger.warning("%s matched no task for %s", label, missing)
