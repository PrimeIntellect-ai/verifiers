"""Shared v1 taskset/harness builders used by the ``vf-eval-v1`` CLI and by
``verifiers.utils.env_utils.load_environment`` when it has to materialize a v1
env from inside a worker subprocess.

Two private kwargs are reserved on ``vf.load_environment(env_id, **env_args)``:

* ``__vf_v1_taskset__`` — dict merged into the taskset's ``TasksetConfig``
  subclass.
* ``__vf_v1_harness__`` — dict describing the harness selection + overrides.
  Recognised shape: ``{"name": "<harness-module>"|None, **overrides}``. The
  harness module is an installed Python module exposing
  ``load_harness(config: HarnessConfig)``.

The CLI emits those when the user customises a v1 env. Workers see the same
``env_args`` and re-run the same builder, so a configured taskset/harness pair
ends up identical in every process.
"""

import inspect
from typing import Any, cast, get_args, get_origin

from pydantic import BaseModel

import verifiers as vf
from verifiers.v1.env import Env as VEnv
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.taskset import Taskset, TasksetConfig

V1_TASKSET_KEY = "__vf_v1_taskset__"
V1_HARNESS_KEY = "__vf_v1_harness__"


def has_v1_overrides(env_args: dict[str, Any]) -> bool:
    """True if ``env_args`` carries the v1 dispatch markers."""
    return V1_TASKSET_KEY in env_args or V1_HARNESS_KEY in env_args


def module_supports_v1_loader(module) -> bool:
    return hasattr(module, "load_taskset")


def _factory_config_type(
    module, factory_name: str, base_type: type[BaseModel]
) -> type | None:
    # Lazy import to avoid pulling env_utils into module init.
    from verifiers.utils.env_utils import factory_config_type

    return factory_config_type(module, factory_name, base_type)


def _import_module_by_name(name: str):
    """Resolve a taskset/harness name to a Python module.

    Names follow the same convention as env ids: dashes become underscores and
    the trailing path segment is used as the module name.
    """
    from verifiers.utils.env_utils import import_env_module

    return import_env_module(name)


def resolve_harness_module(name: str):
    """Import a harness module by name and validate it exposes ``load_harness``."""
    module = _import_module_by_name(name)
    if not hasattr(module, "load_harness"):
        raise AttributeError(
            f"Harness module {module.__name__!r} does not expose "
            "load_harness(config: HarnessConfig). Install a harness package "
            "that provides this factory, or use the env's own load_harness "
            "by omitting the harness positional."
        )
    return module


def harness_config_type_from_module(module) -> type[HarnessConfig]:
    """Inspect a harness module's ``load_harness`` to find its config type."""
    config_type = _factory_config_type(module, "load_harness", HarnessConfig)
    if config_type is None:
        raise TypeError(f"{module.__name__}.load_harness must accept a typed config.")
    return cast(type[HarnessConfig], config_type)


def harness_config_type_from_class(harness_cls: type[Harness]) -> type[HarnessConfig]:
    """Inspect a Harness subclass's ``__init__`` to find its HarnessConfig type."""
    signature = inspect.signature(harness_cls.__init__)
    config_param = signature.parameters.get("config")
    if config_param is None or config_param.annotation is inspect.Parameter.empty:
        return HarnessConfig
    annotation = config_param.annotation
    candidates: list[Any] = (
        list(get_args(annotation))
        if get_origin(annotation) is not None
        else [annotation]
    )
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, HarnessConfig):
            return candidate
    return HarnessConfig


def build_v1_taskset(env_module, overrides: dict[str, Any]) -> Taskset:
    factory = getattr(env_module, "load_taskset", None)
    if factory is None:
        raise AttributeError(
            f"Env module {env_module.__name__!r} does not expose load_taskset; "
            "v1 dispatch is unavailable."
        )
    config_type = _factory_config_type(env_module, "load_taskset", TasksetConfig)
    if config_type is None:
        raise TypeError(
            f"{env_module.__name__}.load_taskset must accept a typed config."
        )
    config = cast(type[TasksetConfig], config_type).model_validate(overrides)
    taskset = factory(config=config)
    if not isinstance(taskset, Taskset):
        raise TypeError(
            f"{env_module.__name__}.load_taskset must return a verifiers.v1.Taskset."
        )
    return taskset


def build_v1_harness(env_module, harness_spec: dict[str, Any]) -> Harness:
    """Resolve and build the v1 harness for an env.

    ``harness_spec`` shape: ``{"name": "<harness-module>"|None, **overrides}``.

    * ``name`` set: import the harness module, read its ``HarnessConfig``
      subclass from ``load_harness``, validate ``overrides`` against it, call
      ``load_harness(config=...)``.
    * ``name`` unset: use the env's own ``load_harness`` if present, else the
      base ``verifiers.v1.Harness`` with ``HarnessConfig`` defaults.
    """
    spec = dict(harness_spec)
    name = spec.pop("name", None)
    if name:
        harness_module = resolve_harness_module(name)
        config_type = harness_config_type_from_module(harness_module)
        config = config_type.model_validate(spec)
        harness = harness_module.load_harness(config=config)
    elif hasattr(env_module, "load_harness"):
        factory_type = _factory_config_type(env_module, "load_harness", HarnessConfig)
        config_type = cast(type[HarnessConfig], factory_type or HarnessConfig)
        config = config_type.model_validate(spec)
        harness = env_module.load_harness(config=config)
    else:
        config = HarnessConfig.model_validate(spec)
        harness = Harness(config=config)
    if not isinstance(harness, Harness):
        raise TypeError(
            f"Resolved harness for env {env_module.__name__!r} is not a "
            "verifiers.v1.Harness instance."
        )
    return harness


def build_v1_env(
    env_id: str,
    *,
    taskset_overrides: dict[str, Any] | None = None,
    harness_spec: dict[str, Any] | None = None,
) -> vf.Environment:
    """Build a ``vf.Env`` for ``env_id`` using the v1 taskset/harness path."""
    env_module = _import_module_by_name(env_id)
    if not module_supports_v1_loader(env_module):
        raise AttributeError(
            f"Env {env_id!r} does not expose load_taskset; v1 dispatch is "
            "only available for taskset/harness environments."
        )
    taskset = build_v1_taskset(env_module, taskset_overrides or {})
    harness = build_v1_harness(env_module, harness_spec or {})
    env = VEnv(taskset=taskset, harness=harness)
    env.env_id = env_id
    return env


def pop_v1_overrides(
    env_args: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the v1 markers from ``env_args``, mutating it in place."""
    taskset_overrides = env_args.pop(V1_TASKSET_KEY, None) or {}
    harness_spec = env_args.pop(V1_HARNESS_KEY, None) or {}
    if not isinstance(taskset_overrides, dict):
        raise TypeError(
            f"{V1_TASKSET_KEY} must be a dict, got {type(taskset_overrides)}."
        )
    if not isinstance(harness_spec, dict):
        raise TypeError(f"{V1_HARNESS_KEY} must be a dict, got {type(harness_spec)}.")
    return taskset_overrides, harness_spec


__all__ = [
    "V1_HARNESS_KEY",
    "V1_TASKSET_KEY",
    "build_v1_env",
    "build_v1_harness",
    "build_v1_taskset",
    "harness_config_type_from_class",
    "harness_config_type_from_module",
    "has_v1_overrides",
    "module_supports_v1_loader",
    "pop_v1_overrides",
    "resolve_harness_module",
]
