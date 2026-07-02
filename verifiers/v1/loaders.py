"""Loaders: resolve a plugin id to its taskset, harness, or judge.

A plugin (taskset, harness, or judge) is a module that exports its `Taskset` / `Harness` /
`Judge` subclass via `__all__` — vf walks the exported names and finds the single subclass of
the base. An id (an `ID`) resolves to that module: a built-in id (`default`, `rlm`,
`harbor`, `binary`, ...) resolves to its namespaced module under the group package
(`verifiers.v1.harnesses.rlm`, `verifiers.v1.tasksets.harbor`, `verifiers.v1.judges.binary`,
...); any other id names a flat module — a local package (hyphens → underscores), or an
`org/name[@version]` package installed on demand from the Environments Hub. Built-ins ship
with verifiers under `verifiers/v1/{harnesses,tasksets,judges}`; custom ones live under
`environments/`, on `sys.path`, or on the hub.

The taskset/harness class carries its types as generic args — `Taskset[TaskT, ConfigT]`,
`Harness[ConfigT]` — which the CLI reads to narrow the plugin's config for `--taskset.*` /
`--harness.*` flags (`taskset_config_type` / `harness_config_type`) and to type the wire trace
(`task_type`).
"""

import importlib
import importlib.util
from types import ModuleType
from typing import Callable, get_args, get_origin

from pydantic_config import BaseConfig

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.judge import Judge, JudgeConfig, judge_config_cls
from verifiers.v1.utils.install import ensure_installed
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig


def narrow_plugin_field(
    data: dict,
    field: str,
    resolve: Callable[[str], type],
    default_id: str | None = None,
) -> None:
    """Narrow `data[field]` (a plugin config, as a dict or BaseConfig) in place to the specific
    config type its `id` resolves to via `resolve` (`taskset_config_type` / `harness_config_type`),
    so plugin-specific fields validate against the real config instead of an untyped dict.
    `default_id` supplies the id when the field omits one (the harness default); a no-op when
    neither is set. Shared by `EnvConfig._resolve_plugins` and `ValidateConfig`."""
    raw = data.get(field)
    if isinstance(raw, BaseConfig):
        raw = raw.model_dump()
    raw = dict(raw or {})
    ident = raw.get("id") or default_id
    if ident:
        data[field] = resolve(ident).model_validate({**raw, "id": ident})


def _import_plugin(plugin_id: str, kind: str, group: str) -> ModuleType:
    """Import a plugin by id. A built-in id resolves to its namespaced module under the
    `group` package (`verifiers.v1.harnesses` / `verifiers.v1.tasksets`); a hub
    `org/name[@version]` id is installed on demand; any other is a local package
    (hyphens → underscores)."""
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
    """The single class exported via `module.__all__` that subclasses `base`. A taskset/harness
    module exports exactly one such class — the walk filters its public names down to subclasses
    of the base, and rejects anything other than exactly one with an informative error."""
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


def _generic_args(cls: type, base: type) -> list:
    """Type args of every `base[...]` specialization across `cls`'s MRO, most-derived first —
    so a subclass that re-binds a type param (e.g. a thin wrapper narrowing the config) wins
    over the base it inherits from."""
    args: list = []
    for klass in cls.__mro__:
        for orig in getattr(klass, "__orig_bases__", ()):
            if get_origin(orig) is base:
                args.extend(get_args(orig))
    return args


def import_taskset(taskset_id: str) -> ModuleType:
    return _import_plugin(taskset_id, "taskset", "verifiers.v1.tasksets")


def import_harness(harness_id: str) -> ModuleType:
    return _import_plugin(harness_id, "harness", "verifiers.v1.harnesses")


def import_judge(judge_id: str) -> ModuleType:
    return _import_plugin(judge_id, "judge", "verifiers.v1.judges")


def taskset_class(taskset_id: str) -> type[Taskset]:
    """The taskset's `Taskset` subclass, exported via its module's `__all__`."""
    return _plugin_class(import_taskset(taskset_id), Taskset, "taskset")


def harness_class(harness_id: str) -> type[Harness]:
    """The harness's `Harness` subclass, exported via its module's `__all__`."""
    return _plugin_class(import_harness(harness_id), Harness, "harness")


def judge_class(judge_id: str) -> type[Judge]:
    """The judge's `Judge` subclass, exported via its module's `__all__`."""
    return _plugin_class(import_judge(judge_id), Judge, "judge")


def default_harness_id(taskset_id: str) -> str:
    """The harness id to use when none is given. A taskset that bundles its own harness — its
    module also exports a `Harness` subclass via `__all__`, so the taskset id doubles as the
    harness id — runs with that harness by default; otherwise the shared `default` harness (a
    bash + edit agent). An explicit `--harness.id` / toml id always takes precedence (this only
    supplies the fallback); for a tool-less chat loop, pass `--harness.id null`."""
    if not taskset_id:
        return "default"
    try:
        module = import_taskset(taskset_id)
        _plugin_class(module, Harness, "harness")
    except (ModuleNotFoundError, TypeError, AttributeError):
        return "default"
    return taskset_id


def load_taskset(config: TasksetConfig) -> Taskset:
    """Build the taskset for a config by dispatching on its `id` (the taskset id)."""
    return taskset_class(config.id)(config)


def load_harness(config: HarnessConfig) -> Harness:
    """Build the harness for a config by dispatching on its `id` (the harness id)."""
    return harness_class(config.id)(config)


def load_judge(config: JudgeConfig) -> Judge:
    """Build a plugged judge for a config by dispatching on its `id` (the judge id)."""
    return judge_class(config.id)(config)


def taskset_config_type(taskset_id: str) -> type[TasksetConfig]:
    """The taskset's `TasksetConfig` subclass, from its `Taskset[TaskT, ConfigT]` generic."""
    for arg in _generic_args(taskset_class(taskset_id), Taskset):
        if isinstance(arg, type) and issubclass(arg, TasksetConfig):
            return arg
    return TasksetConfig


def harness_config_type(harness_id: str) -> type[HarnessConfig]:
    """The harness's config subclass, from its `Harness[ConfigT]` generic."""
    for arg in _generic_args(harness_class(harness_id), Harness):
        if isinstance(arg, type) and issubclass(arg, HarnessConfig):
            return arg
    return HarnessConfig


def judge_config_type(judge_id: str) -> type[JudgeConfig]:
    """The judge's config subclass, from its `Judge[ParsedT, ConfigT]` generic."""
    return judge_config_cls(judge_class(judge_id))


def task_type(taskset_id: str) -> type[Task]:
    """The taskset's `Task` subclass, read off its `Taskset[TaskT, ConfigT]` generic — no data
    is loaded, so a caller that imports the taskset can cheaply build a typed `Trace[TaskT]` for
    the otherwise taskset-specific (loose dict) wire trace. Falls back to the base `Task` when
    no subclass is given."""
    for arg in _generic_args(taskset_class(taskset_id), Taskset):
        if isinstance(arg, type) and issubclass(arg, Task):
            return arg
    return Task
