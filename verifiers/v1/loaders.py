"""Loaders: resolve a plugin id to its taskset or harness.

A plugin (taskset or harness) exposes a single load hook — `load_taskset(config) -> Taskset`
for tasksets, `load_harness(config) -> Harness` for harnesses. An id (an `EnvId`) resolves to
the module exposing it: a built-in id (`default`, `rlm`, `harbor_v1`, `textarena_v1`) resolves
to its namespaced module under the group package (`harnesses.rlm`, `tasksets.harbor_v1`, ...);
any other id names a flat module — a local package (hyphens → underscores), or an
`org/name[@version]` package installed on demand from the Environments Hub.
Built-ins live under `packages/`, installed by default via the `tasksets`/`harnesses` extras;
custom ones live under `examples/`, on `sys.path`, or on the hub. The CLI introspects the
hook's parameter annotation to narrow the plugin's config for `--taskset.*` /
`--harness.*` flags; `task_type` reads the return annotation.
"""

import importlib
import importlib.util
import inspect
from types import ModuleType
from typing import get_args

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.ids import ensure_installed
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig


def _import_plugin(plugin_id: str, kind: str, group: str) -> ModuleType:
    """Import a plugin by id. A built-in id resolves to its namespaced module under the
    `group` package (`harnesses` / `tasksets`); a hub `org/name[@version]` id is installed on
    demand; any other is a local package (hyphens → underscores)."""
    module = ensure_installed(plugin_id)
    namespaced = f"{group}.{module}"
    target = namespaced if importlib.util.find_spec(namespaced) else module
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{kind} {plugin_id!r} not found (tried to import {target!r}). A {kind} is a "
            f"package exposing load_{kind}(config) — the built-in ones are bundled in the "
            f"`{group}` package (vendored by default), installed from the Environments Hub "
            f"(`org/name`), or authored yourself."
        ) from e


def _config_type(load_fn, default: type) -> type:
    """The plugin's config type, read off its load hook's single parameter annotation."""
    param = next(iter(inspect.signature(load_fn).parameters.values()))
    return (
        param.annotation if param.annotation is not inspect.Parameter.empty else default
    )


def import_taskset(taskset_id: str) -> ModuleType:
    return _import_plugin(taskset_id, "taskset", "tasksets")


def import_harness(harness_id: str) -> ModuleType:
    return _import_plugin(harness_id, "harness", "harnesses")


def load_taskset(config: TasksetConfig) -> Taskset:
    """Build the taskset for a config by dispatching on its `id` (the taskset id)."""
    return import_taskset(config.id).load_taskset(config)


def load_harness(config: HarnessConfig) -> Harness:
    """Build the harness for a config by dispatching on its `id` (the harness id)."""
    return import_harness(config.id).load_harness(config)


def taskset_config_type(taskset_id: str) -> type[TasksetConfig]:
    """The taskset's `TasksetConfig` subclass, from `load_taskset`'s parameter annotation."""
    return _config_type(import_taskset(taskset_id).load_taskset, TasksetConfig)


def harness_config_type(harness_id: str) -> type[HarnessConfig]:
    """The harness's config subclass, from `load_harness`'s parameter annotation."""
    return _config_type(import_harness(harness_id).load_harness, HarnessConfig)


def task_type(taskset_id: str) -> type[Task]:
    """The taskset's `Task` subclass, read off `load_taskset`'s return annotation
    (`Taskset[TaskT, ConfigT]`) — no data is loaded, so a caller that imports the taskset
    can cheaply build a typed `Trace[TaskT]` for the otherwise taskset-specific (loose
    dict) wire trace. Falls back to the base `Task` when no subclass is annotated."""
    taskset_type = inspect.signature(
        import_taskset(taskset_id).load_taskset
    ).return_annotation
    candidates = list(get_args(taskset_type))
    for base in getattr(taskset_type, "__orig_bases__", ()):
        candidates += get_args(base)
    for arg in candidates:
        if isinstance(arg, type) and issubclass(arg, Task):
            return arg
    return Task
