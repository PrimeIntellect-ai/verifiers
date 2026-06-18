"""Loaders: resolve a plugin id to its taskset or harness.

A plugin (taskset or harness) is a module that exports its `Taskset` / `Harness` subclass via
`__all__` — vf walks the exported names and finds the single subclass of the base. An id (an
`EnvId`) resolves to that module: a built-in id (`default`, `rlm`, `harbor_v1`, `textarena_v1`)
resolves to its namespaced module under the group package (`harnesses.rlm`, `tasksets.harbor_v1`,
...); any other id names a flat module — a local package (hyphens → underscores), or an
`org/name[@version]` package installed on demand from the Environments Hub.
Built-ins live under `packages/`, installed by default via the `tasksets`/`harnesses` extras;
custom ones live under `environments/`, on `sys.path`, or on the hub.

The taskset/harness class carries its types as generic args — `Taskset[TaskT, ConfigT]`,
`Harness[ConfigT]` — which the CLI reads to narrow the plugin's config for `--taskset.*` /
`--harness.*` flags (`taskset_config_type` / `harness_config_type`) and to type the wire trace
(`task_type`).
"""

import importlib
import importlib.util
import inspect
from types import ModuleType
from typing import Callable, get_args, get_origin, get_type_hints

from pydantic_config import BaseConfig

from verifiers.v1.harness import Harness, HarnessConfig
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
            f"package exporting its {kind.capitalize()} subclass via `__all__` — the built-in "
            f"ones are bundled in the `{group}` package (vendored by default), installed from "
            f"the Environments Hub (`org/name`), or authored yourself."
        ) from e


def _plugin_class(module: ModuleType, base: type, kind: str) -> type | None:
    """The single class exported via `module.__all__` that subclasses `base`. A taskset/harness
    module exports exactly one such class — the walk filters its public names down to subclasses
    of the base, and rejects anything other than exactly one with an informative error."""
    names = getattr(module, "__all__", None)
    if names is None:
        return None
    matches = [
        obj
        for name in names
        if isinstance(obj := getattr(module, name, None), type)
        and issubclass(obj, base)
        and obj is not base
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise TypeError(
            f"{kind} module {module.__name__!r} exports {len(matches)} {base.__name__} "
            f"subclasses via `__all__` ({[c.__name__ for c in matches]}); export exactly one."
        )
    return matches[0]


def _require_plugin_class(module: ModuleType, base: type, kind: str) -> type:
    cls = _plugin_class(module, base, kind)
    if cls is not None:
        return cls
    hook = getattr(module, f"load_{kind}", None)
    if callable(hook):
        raise TypeError(
            f"{kind} module {module.__name__!r} uses legacy load_{kind}(config); "
            "the class is not available for direct introspection. Export the "
            f"{base.__name__} subclass via `__all__` to use this helper."
        )
    names = getattr(module, "__all__", None)
    if names is None:
        raise AttributeError(
            f"{kind} module {module.__name__!r} defines no `__all__`; a {kind} must export its "
            f"{base.__name__} subclass via `__all__` (e.g. `__all__ = ['My{base.__name__}']`)."
        )
    raise TypeError(
        f"{kind} module {module.__name__!r} exports no {base.__name__} subclass via "
        f"`__all__` (found {list(names)}); export exactly one."
    )


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


def _config_type(load_fn: Callable, default: type) -> type:
    """Legacy hook config type, read from its single parameter annotation."""
    try:
        param = next(iter(inspect.signature(load_fn).parameters.values()))
    except StopIteration:
        return default
    if param.annotation is inspect.Parameter.empty:
        return default
    return get_type_hints(load_fn).get(param.name, param.annotation)


def _return_args(load_fn: Callable) -> list:
    """Generic args from a legacy hook return annotation."""
    return_type = get_type_hints(load_fn).get(
        "return", inspect.signature(load_fn).return_annotation
    )
    candidates = list(get_args(return_type))
    for base in getattr(return_type, "__orig_bases__", ()):
        candidates += get_args(base)
    return candidates


def import_taskset(taskset_id: str) -> ModuleType:
    return _import_plugin(taskset_id, "taskset", "tasksets")


def import_harness(harness_id: str) -> ModuleType:
    return _import_plugin(harness_id, "harness", "harnesses")


def taskset_class(taskset_id: str) -> type[Taskset]:
    """The taskset's `Taskset` subclass, exported via its module's `__all__`."""
    return _require_plugin_class(import_taskset(taskset_id), Taskset, "taskset")


def harness_class(harness_id: str) -> type[Harness]:
    """The harness's `Harness` subclass, exported via its module's `__all__`."""
    return _require_plugin_class(import_harness(harness_id), Harness, "harness")


def load_taskset(config: TasksetConfig) -> Taskset:
    """Build the taskset for a config by dispatching on its `id` (the taskset id)."""
    module = import_taskset(config.id)
    cls = _plugin_class(module, Taskset, "taskset")
    if cls is not None:
        return cls(config)
    load_fn = getattr(module, "load_taskset", None)
    if callable(load_fn):
        return load_fn(config)
    return taskset_class(config.id)(config)


def load_harness(config: HarnessConfig) -> Harness:
    """Build the harness for a config by dispatching on its `id` (the harness id)."""
    module = import_harness(config.id)
    cls = _plugin_class(module, Harness, "harness")
    if cls is not None:
        return cls(config)
    load_fn = getattr(module, "load_harness", None)
    if callable(load_fn):
        return load_fn(config)
    return harness_class(config.id)(config)


def taskset_config_type(taskset_id: str) -> type[TasksetConfig]:
    """The taskset config subclass, from class generics or a legacy hook annotation."""
    module = import_taskset(taskset_id)
    cls = _plugin_class(module, Taskset, "taskset")
    if cls is not None:
        for arg in _generic_args(cls, Taskset):
            if isinstance(arg, type) and issubclass(arg, TasksetConfig):
                return arg
        return TasksetConfig
    load_fn = getattr(module, "load_taskset", None)
    if callable(load_fn):
        typ = _config_type(load_fn, TasksetConfig)
        if isinstance(typ, type) and issubclass(typ, TasksetConfig):
            return typ
    return TasksetConfig


def harness_config_type(harness_id: str) -> type[HarnessConfig]:
    """The harness config subclass, from class generics or a legacy hook annotation."""
    module = import_harness(harness_id)
    cls = _plugin_class(module, Harness, "harness")
    if cls is not None:
        for arg in _generic_args(cls, Harness):
            if isinstance(arg, type) and issubclass(arg, HarnessConfig):
                return arg
        return HarnessConfig
    load_fn = getattr(module, "load_harness", None)
    if callable(load_fn):
        typ = _config_type(load_fn, HarnessConfig)
        if isinstance(typ, type) and issubclass(typ, HarnessConfig):
            return typ
    return HarnessConfig


def task_type(taskset_id: str) -> type[Task]:
    """The taskset's `Task` subclass, from class generics or a legacy hook."""
    module = import_taskset(taskset_id)
    cls = _plugin_class(module, Taskset, "taskset")
    candidates = _generic_args(cls, Taskset) if cls is not None else []
    load_fn = getattr(module, "load_taskset", None)
    if not candidates and callable(load_fn):
        candidates = _return_args(load_fn)
    for arg in candidates:
        if isinstance(arg, type) and issubclass(arg, Task):
            return arg
    return Task
