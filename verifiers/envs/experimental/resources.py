from __future__ import annotations

from asyncio import Lock
from collections.abc import AsyncIterator
from collections.abc import Callable
from collections.abc import Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar, cast

from verifiers.clients import Client
from verifiers.rubrics.rubric import Rubric
from verifiers.types import DatasetBuilder, SamplingArgs

from verifiers.envs.experimental.channels.channel import (
    lifecycle_handlers,
    resolve_resource_objects,
)
from verifiers.envs.experimental.channels import (
    Endpoint,
    SandboxResources,
    ToolMonitorRubric,
    ToolRegistry,
    User,
    attach_resources,
    compose_rubrics,
)
from verifiers.envs.experimental.task import Task
from verifiers.envs.experimental.taskset import Taskset

if TYPE_CHECKING:
    from datasets import Dataset

    from verifiers.envs.experimental.harness import Harness

LifecycleHandler = Callable[..., object]
T = TypeVar("T")


@dataclass
class Resources:
    """Environment-scope resolved objects shared across rollouts."""

    taskset: Taskset
    harness: Harness
    objects: dict[str, object] = field(default_factory=dict)
    stop_conditions: list[LifecycleHandler] = field(default_factory=list)
    cleanup_handlers: list[LifecycleHandler] = field(default_factory=list)
    teardown_handlers: list[LifecycleHandler] = field(default_factory=list)
    dataset: Dataset | DatasetBuilder | None = field(default=None, init=False)
    eval_dataset: Dataset | DatasetBuilder | None = field(default=None, init=False)
    env_id: str | None = field(default=None, init=False)
    rubric: Rubric = field(init=False)
    tools: ToolRegistry = field(init=False)
    _teardown_complete: bool = False
    _rollout_objects: ContextVar[dict[str, object] | None] = field(
        default_factory=lambda: ContextVar("vf_rollout_objects", default=None),
        init=False,
        repr=False,
    )
    _client: ContextVar[Client | None] = field(
        default_factory=lambda: ContextVar("vf_rollout_client", default=None),
        init=False,
        repr=False,
    )
    _model: ContextVar[str | None] = field(
        default_factory=lambda: ContextVar("vf_rollout_model", default=None),
        init=False,
        repr=False,
    )
    _sampling_args: ContextVar[SamplingArgs | None] = field(
        default_factory=lambda: ContextVar("vf_rollout_sampling_args", default=None),
        init=False,
        repr=False,
    )
    _runtime: ContextVar[dict[str, object] | None] = field(
        default_factory=lambda: ContextVar("vf_rollout_runtime", default=None),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        resolution = resolve_resource_objects(self.taskset, self.harness, phase="env")
        self.dataset = (
            cast(DatasetBuilder, self.taskset.get_dataset)
            if self.taskset.has_dataset()
            else None
        )
        self.eval_dataset = (
            cast(DatasetBuilder, self.taskset.get_eval_dataset)
            if self.taskset.has_eval_dataset()
            else None
        )
        self.env_id = self.taskset.name
        self.objects.update(resolution.objects)
        self.tools = self.require("tools", ToolRegistry)
        tool_names = self.tools.names()
        self.rubric = compose_rubrics(
            self.require("rubric", Rubric),
            ToolMonitorRubric(tool_names=tool_names) if tool_names else None,
        )
        self.objects["rubric"] = self.rubric
        self._extend_handlers("teardown", [self.tools.teardown])
        self._extend_handlers("stop", resolution.stop_conditions)
        self._extend_handlers("cleanup", resolution.cleanup_handlers)
        self._extend_handlers("teardown", resolution.teardown_handlers)
        self.harness.resources = self
        attach_resources(self.rubric, self)

    def __getattr__(self, name: str) -> object:
        rollout_objects = self.rollout_objects()
        if name in rollout_objects:
            return rollout_objects[name]
        try:
            return self.objects[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: object) -> None:
        if name in self.__dataclass_fields__ or "objects" not in self.__dict__:
            super().__setattr__(name, value)
            return
        self.objects[name] = value

    def get(self, name: str, default: object = None) -> object:
        rollout_objects = self.rollout_objects()
        if name in rollout_objects:
            return rollout_objects[name]
        return self.objects.get(name, default)

    def get_global(self, name: str, default: object = None) -> object:
        return self.objects.get(name, default)

    def require(self, name: str, expected_type: type[T]) -> T:
        value = self.get(name)
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Resolved resource {name!r} must be {expected_type.__name__}."
            )
        return value

    @property
    def system_prompt(self) -> str:
        return cast(str, self.require("system_prompt", str))

    @property
    def user(self) -> User | None:
        value = self.get("user")
        if value is None:
            return None
        if not isinstance(value, User):
            raise TypeError("Resolved resource 'user' must implement User.")
        return value

    @property
    def endpoint(self) -> Endpoint | None:
        value = self.get("endpoint")
        if value is None:
            return None
        if not isinstance(value, Endpoint):
            raise TypeError("Resolved resource 'endpoint' must be Endpoint.")
        return value

    @property
    def sandbox_runtime(self) -> SandboxResources | None:
        value = self.get("sandbox_runtime")
        if value is None:
            return None
        if not isinstance(value, SandboxResources):
            raise TypeError(
                "Resolved resource 'sandbox_runtime' must be SandboxResources."
            )
        return value

    @property
    def sandbox_request(self) -> object:
        return self.get("sandbox_request")

    @property
    def sandbox_scoring(self) -> bool:
        return bool(self.get("sandbox_scoring"))

    @property
    def upload_dirs(self) -> Mapping[str, object]:
        value = self.get("upload_dirs")
        if not isinstance(value, Mapping):
            raise TypeError("Resolved resource 'upload_dirs' must be a mapping.")
        return cast(Mapping[str, object], value)

    def rollout_objects(self) -> dict[str, object]:
        return self._rollout_objects.get() or {}

    @property
    def client(self) -> Client:
        client = self._client.get()
        if client is None:
            raise RuntimeError("No model client is active for this rollout.")
        return client

    @property
    def model(self) -> str:
        model = self._model.get()
        if model is None:
            raise RuntimeError("No model is active for this rollout.")
        return model

    @property
    def sampling_args(self) -> SamplingArgs:
        return self._sampling_args.get() or {}

    @property
    def runtime(self) -> dict[str, object]:
        runtime = self._runtime.get()
        if runtime is None:
            raise RuntimeError("No rollout runtime is active.")
        return runtime

    def _extend_handlers(self, kind: str, handlers: object) -> None:
        resolved = lifecycle_handlers(handlers)
        if kind == "stop":
            self.stop_conditions.extend(resolved)
        elif kind == "cleanup":
            self.cleanup_handlers.extend(resolved)
        elif kind == "teardown":
            self.teardown_handlers.extend(resolved)
        else:
            raise ValueError(f"Unknown lifecycle handler kind: {kind}")

    def current_handlers(self, kind: str) -> list[LifecycleHandler]:
        if kind == "stop":
            base = self.stop_conditions
            runtime_key = "stop_conditions"
        elif kind == "cleanup":
            base = self.cleanup_handlers
            runtime_key = "cleanup_handlers"
        elif kind == "teardown":
            base = self.teardown_handlers
            runtime_key = "teardown_handlers"
        else:
            raise ValueError(f"Unknown lifecycle handler kind: {kind}")
        runtime = self._runtime.get() or {}
        local = lifecycle_handlers(runtime.get(runtime_key))
        return unique_handlers([*base, *local])

    @asynccontextmanager
    async def rollout(
        self,
        task: Task,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> AsyncIterator[None]:
        resolution = resolve_resource_objects(
            self.taskset,
            self.harness,
            phase="rollout",
            task=task,
            normalize=False,
        )
        stop_token = self._rollout_objects.set(resolution.objects)
        client_token = self._client.set(client)
        model_token = self._model.set(model)
        sampling_token = self._sampling_args.set(sampling_args or {})
        runtime_token = self._runtime.set(
            {
                "stop_conditions": resolution.stop_conditions,
                "cleanup_handlers": resolution.cleanup_handlers,
                "teardown_handlers": resolution.teardown_handlers,
            }
        )
        try:
            yield
        finally:
            self._runtime.reset(runtime_token)
            self._sampling_args.reset(sampling_token)
            self._model.reset(model_token)
            self._client.reset(client_token)
            self._rollout_objects.reset(stop_token)

    def trajectory_lock(self) -> Lock:
        lock = self.runtime.setdefault("trajectory_lock", Lock())
        return cast(Lock, lock)

    async def teardown(self) -> None:
        if self._teardown_complete:
            return
        self._teardown_complete = True
        await self.harness.run_teardown_handlers(self.teardown_handlers)


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
