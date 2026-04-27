from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

BindingResolver = Callable[[object], object]
BindingRef = str | BindingResolver


@dataclass(frozen=True)
class BindingContext:
    resources: object
    task: object
    state: object


@dataclass(frozen=True)
class Binding:
    resolve: Callable[[BindingContext], object]


@dataclass(frozen=True)
class StateBinding:
    ref: BindingRef

    def resolve(self, context: BindingContext) -> object:
        return scoped_binding_value(context.state, self.ref, "state")


@dataclass(frozen=True)
class TaskBinding:
    ref: BindingRef

    def resolve(self, context: BindingContext) -> object:
        if isinstance(self.ref, str):
            return task_binding_value(context.task, self.ref)
        return self.ref(context.task)


@dataclass(frozen=True)
class ResourceBinding:
    ref: BindingRef

    def resolve(self, context: BindingContext) -> object:
        if not isinstance(self.ref, str):
            return self.ref(context.resources)
        require = getattr(context.resources, "require", None)
        if not callable(require):
            raise TypeError("resources must provide require() for ResourceBinding.")
        return require(self.ref)


def scoped_binding_value(source: object, ref: BindingRef, source_name: str) -> object:
    if not isinstance(ref, str):
        return ref(source)
    if not isinstance(source, Mapping):
        raise TypeError(f"{source_name} must be a mapping for binding {ref!r}.")
    mapping = cast(Mapping[str, object], source)
    if ref not in mapping:
        raise KeyError(f"{source_name} does not contain binding key {ref!r}.")
    return mapping[ref]


def task_binding_value(task: object, key: str) -> object:
    if isinstance(task, Mapping) and key in task:
        return cast(Mapping[str, object], task)[key]
    if hasattr(task, key):
        return getattr(task, key)
    raise KeyError(f"task does not contain binding key {key!r}.")
