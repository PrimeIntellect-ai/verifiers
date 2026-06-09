"""Loaders: resolve a plugin id to its taskset or harness.

A plugin (taskset or harness) is just a Python package named by its id, depending on
v1 and exposing a single load hook — `load_taskset(config) -> Taskset` for tasksets,
`load_harness(config) -> Harness` for harnesses. Resolution is the same for every plugin
(no registry, no built-in special case): the id is imported as a module (hyphens →
underscores). The shipped plugins (`harbor`, `default`, `rlm`) are ordinary packages under
`packages/`, installed by default via the `tasksets`/`harnesses` extras; custom ones live
under `examples/` or anywhere on `sys.path`. The CLI introspects the hook's parameter
annotation to narrow the plugin's config for `--taskset.*` / `--harness.*` flags;
`task_type` reads the return annotation.
"""

import importlib
import inspect
from types import ModuleType
from typing import get_args

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig


def _import_plugin(plugin_id: str, kind: str, group: str) -> ModuleType:
    """Import a plugin by id — the id is the module name (hyphens → underscores)."""
    module = plugin_id.replace("-", "_")
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{kind} {plugin_id!r} not found (tried to import {module!r}). A {kind} is a package "
            f"exposing load_{kind}(config) — the shipped ones are bundled in the `{group}` "
            f"package (vendored by default), or install/author your own."
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
