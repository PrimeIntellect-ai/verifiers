from __future__ import annotations

from asyncio import Lock
from collections.abc import AsyncIterator
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeVar, cast, overload

from verifiers.clients import Client
from verifiers.rubrics.rubric import Rubric
from verifiers.types import DatasetBuilder, SamplingArgs
from verifiers.utils.async_utils import maybe_await

from verifiers.envs.experimental.channels.channel import (
    LifecycleHooks,
    ResourceType,
    channel_definitions,
    channel_resource_types,
    resolve_resource_objects,
)
from verifiers.envs.experimental.channels import (
    ToolRegistry,
    attach_resources,
)
from verifiers.envs.experimental.task import Task
from verifiers.envs.experimental.taskset import Taskset

if TYPE_CHECKING:
    from datasets import Dataset

    from verifiers.envs.experimental.harness import Harness

LifecycleHandler = Callable[..., object]
ScoringMode = Literal["run", "group", "none"]
T = TypeVar("T")


@dataclass(frozen=True)
class _RolloutState:
    objects: dict[str, object]
    client: Client
    model: str
    sampling_args: SamplingArgs
    hooks: LifecycleHooks
    runtime: dict[str, object]


@dataclass
class Resources:
    """Environment-scope resolved objects shared across rollouts."""

    taskset: Taskset | None = None
    harness: Harness | None = None
    scoring: ScoringMode | None = None
    objects: dict[str, object] = field(default_factory=dict)
    resource_types: dict[str, ResourceType] = field(default_factory=dict)
    hooks: LifecycleHooks = field(default_factory=LifecycleHooks)
    dataset: Dataset | DatasetBuilder | None = field(default=None, init=False)
    eval_dataset: Dataset | DatasetBuilder | None = field(default=None, init=False)
    env_id: str | None = field(default=None, init=False)
    _teardown_complete: bool = False
    _rollout: ContextVar[_RolloutState | None] = field(
        default_factory=lambda: ContextVar("vf_rollout", default=None),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.harness is None:
            raise ValueError("Resources requires a harness.")
        if self.scoring is None:
            self.scoring = "group" if self.taskset is not None else "run"
        if self.scoring not in {"run", "group", "none"}:
            raise ValueError("resources.scoring must be 'run', 'group', or 'none'.")
        owners = self.owners()
        definitions = channel_definitions(*owners)
        resolution = resolve_resource_objects(*owners, phase="env")
        self.dataset = (
            cast(DatasetBuilder, self.taskset.get_dataset)
            if self.taskset is not None and self.taskset.has_dataset()
            else None
        )
        self.eval_dataset = (
            cast(DatasetBuilder, self.taskset.get_eval_dataset)
            if self.taskset is not None and self.taskset.has_eval_dataset()
            else None
        )
        self.env_id = self.taskset.name if self.taskset is not None else None
        self.objects.update(resolution.objects)
        self.resource_types.update(channel_resource_types(definitions))
        rubric = self.require("rubric", Rubric)
        attach_resources(rubric, self)
        self.hooks = resolution.hooks
        self.harness.resources = self

    def owners(self) -> tuple[object, ...]:
        owners: list[object] = []
        if self.taskset is not None:
            owners.append(self.taskset)
        if self.harness is not None:
            owners.append(self.harness)
        return tuple(owners)

    def __getattr__(self, name: str) -> object:
        rollout_objects = self.rollout_objects()
        if name in rollout_objects:
            return rollout_objects[name]
        try:
            return self.objects[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def get(self, name: str, default: object = None) -> object:
        rollout_objects = self.rollout_objects()
        if name in rollout_objects:
            return rollout_objects[name]
        return self.objects.get(name, default)

    @property
    def rubric(self) -> Rubric:
        return self.require("rubric", Rubric)

    @property
    def tools(self) -> ToolRegistry:
        return self.require("tools", ToolRegistry)

    @overload
    def require(self, name: str, expected_type: type[T]) -> T: ...

    @overload
    def require(self, name: str) -> object: ...

    def require(self, name: str, expected_type: type[T] | None = None) -> T | object:
        if name not in self.rollout_objects() and name not in self.objects:
            raise KeyError(f"Resolved resource {name!r} is not available.")
        value = self.get(name)
        resource_type = expected_type or self.resource_types.get(name)
        if (
            resource_type is not None
            and resource_type is not object
            and not isinstance(value, resource_type)
        ):
            raise TypeError(
                f"Resolved resource {name!r} must be {resource_type.__name__}."
            )
        return value

    def rollout_objects(self) -> dict[str, object]:
        rollout = self._rollout.get()
        if rollout is None:
            return {}
        return rollout.objects

    @property
    def client(self) -> Client:
        rollout = self._rollout.get()
        if rollout is None:
            raise RuntimeError("No model client is active for this rollout.")
        return rollout.client

    @property
    def model(self) -> str:
        rollout = self._rollout.get()
        if rollout is None:
            raise RuntimeError("No model is active for this rollout.")
        return rollout.model

    @property
    def sampling_args(self) -> SamplingArgs:
        rollout = self._rollout.get()
        if rollout is None:
            return {}
        return rollout.sampling_args

    @property
    def runtime(self) -> dict[str, object]:
        rollout = self._rollout.get()
        if rollout is None:
            raise RuntimeError("No rollout runtime is active.")
        return rollout.runtime

    def current_handlers(self, kind: str) -> list[LifecycleHandler]:
        rollout = self._rollout.get()
        rollout_hooks = LifecycleHooks() if rollout is None else rollout.hooks
        return sorted_handlers(
            kind,
            unique_handlers([*self.hooks.get(kind), *rollout_hooks.get(kind)]),
        )

    async def run_handlers(self, kind: str, *args: object) -> None:
        for handler in self.current_handlers(kind):
            await maybe_await(handler, *args)

    @asynccontextmanager
    async def rollout(
        self,
        task: Task,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> AsyncIterator[None]:
        resolution = resolve_resource_objects(
            *self.owners(), phase="rollout", task=task
        )
        rollout_token = self._rollout.set(
            _RolloutState(
                objects=dict(resolution.objects),
                client=client,
                model=model,
                sampling_args=sampling_args or {},
                hooks=resolution.hooks,
                runtime={},
            )
        )
        try:
            yield
        finally:
            self._rollout.reset(rollout_token)

    def trajectory_lock(self) -> Lock:
        lock = self.runtime.setdefault("trajectory_lock", Lock())
        return cast(Lock, lock)

    async def teardown(self) -> None:
        if self._teardown_complete:
            return
        self._teardown_complete = True
        await self.run_handlers("teardown")


def unique_handlers(handlers: list[LifecycleHandler]) -> list[LifecycleHandler]:
    unique: list[LifecycleHandler] = []
    seen: set[tuple[int, int]] = set()
    for handler in handlers:
        key = (
            id(getattr(handler, "__self__", None)),
            id(getattr(handler, "__func__", handler)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(handler)
    return unique


def sorted_handlers(
    kind: str, handlers: list[LifecycleHandler]
) -> list[LifecycleHandler]:
    priority_attr = f"{kind}_priority"
    return sorted(
        handlers,
        key=lambda handler: (
            -getattr(handler, priority_attr, 0),
            str(getattr(handler, "__name__", "")),
        ),
    )
