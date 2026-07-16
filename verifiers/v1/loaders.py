"""Resolve taskset, harness, judge, and environment plugins."""

import importlib
import importlib.util
from types import ModuleType
from typing import Callable

from pydantic_config import BaseConfig

from verifiers.v1.env import EnvConfig, EnvParams, Environment
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
    namespaced = f"{group}.{module}"
    target = namespaced if importlib.util.find_spec(namespaced) else module
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{kind} {plugin_id!r} not found (tried to import {target!r}). A {kind} is a "
            f"package exporting its {kind.capitalize()} subclass via `__all__` — the built-in "
            f"ones ship with verifiers in the `{group}` package, installed from "
            f"the Environments Hub (`org/name`), or authored yourself."
        ) from e


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
        # ValueError, not TypeError: "no plugin here" (TypeError) is a state the
        # taskset-fallback callers legitimately swallow — an ambiguous export is an
        # authoring error that must stay loud everywhere.
        raise ValueError(
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


def import_environment(env_id: str) -> ModuleType:
    return _import_plugin(env_id, "environment", "verifiers.v1.envs")


def environment_class(taskset_id: str, env_id: str = "") -> type[Environment]:
    """The `Environment` — the control flow between agents — for a run. An explicit
    `env_id` (`--env.id`) names it directly: a bundled env (`verifiers.v1.envs`), a
    local package, or a Hub id — and a failure to resolve it raises (an explicit
    pairing must not silently fall back). Otherwise the taskset's own story: its
    package's `Environment` subclass when it exports one via `__all__` (a recipe env
    ships with its taskset, the same plugin idiom as a bundled harness), else the base
    `Environment` — whose defaults ARE the single-agent case, so every plain taskset
    resolves to today's behavior."""
    if env_id:
        return _plugin_class(import_environment(env_id), Environment, "environment")
    if not taskset_id:
        return Environment
    try:
        module = import_taskset(taskset_id)
        return _plugin_class(module, Environment, "environment")
    except (ModuleNotFoundError, TypeError, AttributeError):
        return Environment


def load_environment(config: EnvConfig) -> Environment:
    """Construct the env for `config`: the `--env.id`-selected `Environment` when set,
    else the taskset's exported subclass when there is one, else the base. Every env
    construction site (eval, serve, gepa) goes through here so subclass envs load
    everywhere."""
    return environment_class(config.taskset.id, config.env.id)(config)


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


def env_params_type(taskset_id: str, env_id: str = "") -> type[EnvParams]:
    """Resolve the env's params specialization (`Environment[YourParams]`) through its
    MRO — keyed by the selected env id when set, else the taskset's own env — the empty
    base `EnvParams` for a plain taskset. `EnvConfig` narrows its `env` field to this,
    which is what gives `--env.<role>.model` CLI/TOML addressing."""
    return (
        generic_type(
            environment_class(taskset_id, env_id), EnvParams, origin=Environment
        )
        or EnvParams
    )


def task_type(taskset_id: str) -> type[Task]:
    """The taskset's `Task` subclass from its `Taskset[TaskT, ConfigT]` generic — no
    data is loaded, so replay can cheaply recover the task data type. Falls back to
    the base `Task` when no subclass is given."""
    return generic_type(taskset_class(taskset_id), Task, origin=Taskset) or Task
