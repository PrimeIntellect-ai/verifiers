"""Resolve taskset, harness, and judge plugins."""

import importlib
from types import ModuleType
from typing import Callable

from pydantic_config import BaseConfig

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.judge import Judge, JudgeConfig, judge_config_cls
from verifiers.v1.utils.install import ensure_installed
from verifiers.v1.utils.generic import generic_type
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig


def narrow_plugin_field(
    data: dict,
    field: str,
    resolve: Callable[[str], type],
    default_id: str | None = None,
) -> None:
    raw = data.get(field)
    if isinstance(raw, BaseConfig):
        raw = raw.model_dump()
    raw = dict(raw or {})
    ident = raw.get("id") or default_id
    if ident:
        data[field] = resolve(ident).model_validate({**raw, "id": ident})


def _import_plugin(plugin_id: str, kind: str, group: str) -> ModuleType:
    module = ensure_installed(plugin_id)
    targets = (f"{group}.{module}", module)
    last_error: ModuleNotFoundError | None = None
    for target in targets:
        try:
            return importlib.import_module(target)
        except ModuleNotFoundError as error:
            # A missing candidate (or one of its parent packages) permits the flat-module
            # fallback. A dependency imported from inside a found plugin does not.
            if error.name is None or not (
                error.name == target or target.startswith(f"{error.name}.")
            ):
                raise
            last_error = error
    tried = ", ".join(repr(target) for target in targets)
    raise ModuleNotFoundError(
        f"{kind} {plugin_id!r} not found (normalized module {module!r}; tried imports: "
        f"{tried}). A {kind} is a package exporting its {kind.capitalize()} subclass via "
        f"`__all__` — the built-in ones ship with verifiers in the `{group}` package, "
        f"installed from the Environments Hub (`org/name`), installed from an external "
        f"package, or authored yourself."
    ) from last_error


def _plugin_class(module: ModuleType, base: type, kind: str) -> type:
    names = getattr(module, "__all__", None)
    if names is None:
        raise AttributeError(
            f"{kind} module {module.__name__!r} defines no `__all__`; a {kind} must export its "
            f"{base.__name__} subclass via `__all__` (e.g. `__all__ = ['My{base.__name__}']`)."
        )
    matches = [
        obj
        for name in names
        if isinstance(obj := getattr(module, name, None), type)
        and issubclass(obj, base)
        and obj is not base
    ]
    if not matches:
        raise TypeError(
            f"{kind} module {module.__name__!r} exports no {base.__name__} subclass via "
            f"`__all__` (found {list(names)}); export exactly one."
        )
    if len(matches) > 1:
        raise TypeError(
            f"{kind} module {module.__name__!r} exports {len(matches)} {base.__name__} "
            f"subclasses via `__all__` ({[c.__name__ for c in matches]}); export exactly one."
        )
    return matches[0]


def import_taskset(taskset_id: str) -> ModuleType:
    return _import_plugin(taskset_id, "taskset", "verifiers.v1.tasksets")


def import_harness(harness_id: str) -> ModuleType:
    return _import_plugin(harness_id, "harness", "verifiers.v1.harnesses")


def import_judge(judge_id: str) -> ModuleType:
    return _import_plugin(judge_id, "judge", "verifiers.v1.judges")


def taskset_class(taskset_id: str) -> type[Taskset]:
    return _plugin_class(import_taskset(taskset_id), Taskset, "taskset")


def harness_class(harness_id: str) -> type[Harness]:
    return _plugin_class(import_harness(harness_id), Harness, "harness")


def judge_class(judge_id: str) -> type[Judge]:
    return _plugin_class(import_judge(judge_id), Judge, "judge")


def default_harness_id(taskset_id: str) -> str:
    if not taskset_id:
        return "default"
    try:
        module = import_taskset(taskset_id)
        _plugin_class(module, Harness, "harness")
    except (ModuleNotFoundError, TypeError, AttributeError):
        return "default"
    return taskset_id


def load_taskset(config: TasksetConfig) -> Taskset:
    return taskset_class(config.id)(config)


def load_harness(config: HarnessConfig) -> Harness:
    return harness_class(config.id)(config)


def load_judge(config: JudgeConfig) -> Judge:
    return judge_class(config.id)(config)


def taskset_config_type(taskset_id: str) -> type[TasksetConfig]:
    """Resolve the taskset's config specialization through its MRO."""
    return (
        generic_type(taskset_class(taskset_id), TasksetConfig, origin=Taskset)
        or TasksetConfig
    )


def harness_config_type(harness_id: str) -> type[HarnessConfig]:
    """Resolve the harness's config specialization through its MRO."""
    return (
        generic_type(harness_class(harness_id), HarnessConfig, origin=Harness)
        or HarnessConfig
    )


def judge_config_type(judge_id: str) -> type[JudgeConfig]:
    """Resolve the judge's config specialization through its MRO."""
    return judge_config_cls(judge_class(judge_id))


def task_type(taskset_id: str) -> type[Task]:
    """The taskset's `Task` subclass from its `Taskset[TaskT, ConfigT]` generic — no
    data is loaded, so replay can cheaply recover the task data type. Falls back to
    the base `Task` when no subclass is given."""
    return generic_type(taskset_class(taskset_id), Task, origin=Taskset) or Task
