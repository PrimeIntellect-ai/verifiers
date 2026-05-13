"""Module loaders for v1 Taskset / Harness packages.

Mirrors the shape of ``verifiers.load_environment(env_id, **env_args)`` for
the v1 split:

    taskset = vf.load_taskset("reverse-text")
    harness = vf.load_harness("opencode")
    env = vf.Env(taskset=taskset, harness=harness)

Each loader imports the named package, inspects its
``load_taskset(config: X)`` / ``load_harness(config: Y)`` signature for the
concrete config subclass, coerces any dict / Config argument into ``X``/``Y``,
and calls the loader.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Mapping
from types import ModuleType
from typing import Any, cast, get_args, get_origin

from .config import HarnessConfig, TasksetConfig
from .harness import Harness
from .taskset import Taskset

logger = logging.getLogger(__name__)


HARNESS_REGISTRY: dict[str, str] = {
    "base": "verifiers.v1.packages.harnesses.base",
    "opencode": "verifiers.v1.packages.harnesses.opencode",
    "rlm": "verifiers.v1.packages.harnesses.rlm",
    "pi": "verifiers.v1.packages.harnesses.pi",
    "mini-swe": "verifiers.v1.packages.harnesses.mini_swe_agent",
}


# --------------------------------------------------------------------------- #
# Module resolution.
# --------------------------------------------------------------------------- #


def import_taskset_module(taskset_id: str) -> ModuleType:
    """Import an env package, preferring the side-by-side ``*_v1`` module."""
    base = taskset_id.replace("-", "_").split("/")[-1]
    last_err: Exception | None = None
    for module_name in (base, f"{base}_v1"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            last_err = e
            continue
        if hasattr(module, "load_taskset"):
            return module
        last_err = AttributeError(f"{module_name!r} has no `load_taskset`")
    raise ValueError(
        f"Could not load taskset {taskset_id!r}. Ensure the package is installed. "
        f"Last error: {last_err}"
    )


def resolve_harness_module(harness_id: str) -> ModuleType:
    """Resolve a harness id to a module exposing ``load_harness``.

    Tries, in order:
      1. The harness registry (``base``/``opencode``/``rlm``/...).
      2. ``harness_id`` itself as a Python module path.
      3. The env-package convention (``<name>`` or ``<name>_v1``), so that
         ``vf.load_harness("reverse-text")`` resolves to ``reverse_text_v1``.
    """
    target = HARNESS_REGISTRY.get(harness_id, harness_id)
    base = harness_id.replace("-", "_").split("/")[-1]
    candidates = [target]
    if base not in candidates:
        candidates.append(base)
    if f"{base}_v1" not in candidates:
        candidates.append(f"{base}_v1")

    last_err: Exception | None = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            last_err = e
            continue
        if hasattr(module, "load_harness"):
            return module
        last_err = AttributeError(f"{module_name!r} has no `load_harness`")
    raise ValueError(
        f"Could not resolve harness {harness_id!r}. Last error: {last_err}"
    )


# --------------------------------------------------------------------------- #
# Config-type discovery from loader signature.
# --------------------------------------------------------------------------- #


def _config_type_from_annotation(annotation: Any, fallback: type) -> type:
    if annotation is inspect.Parameter.empty or annotation is None:
        return fallback
    if get_origin(annotation) is None:
        return annotation if isinstance(annotation, type) else fallback
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        if isinstance(arg, type):
            return arg
    return fallback


def _resolve_config_annotation(fn: Any, fallback: type) -> type:
    try:
        hints = inspect.get_annotations(fn, eval_str=True)
    except Exception:
        hints = {}
    return _config_type_from_annotation(
        hints.get("config", inspect.Parameter.empty), fallback
    )


def _check_single_config_param(fn: Any, fn_name: str) -> None:
    params = list(inspect.signature(fn).parameters.values())
    if len(params) != 1 or params[0].name != "config":
        raise TypeError(
            f"{fn.__module__}.{fn_name} must take exactly one positional "
            f"`config` argument (got {[p.name for p in params]})."
        )


def get_taskset_config_cls(module: ModuleType) -> type[TasksetConfig]:
    _check_single_config_param(module.load_taskset, "load_taskset")
    return cast(
        type[TasksetConfig],
        _resolve_config_annotation(module.load_taskset, TasksetConfig),
    )


def get_harness_config_cls(module: ModuleType) -> type[HarnessConfig]:
    if not hasattr(module, "load_harness"):
        return HarnessConfig
    _check_single_config_param(module.load_harness, "load_harness")
    return cast(
        type[HarnessConfig],
        _resolve_config_annotation(module.load_harness, HarnessConfig),
    )


# --------------------------------------------------------------------------- #
# Public loaders.
# --------------------------------------------------------------------------- #


def _coerce_config(
    raw: TasksetConfig | HarnessConfig | Mapping[str, Any] | None,
    config_cls: type[TasksetConfig] | type[HarnessConfig],
) -> Any:
    if raw is None or isinstance(raw, config_cls):
        return raw
    if isinstance(raw, (TasksetConfig, HarnessConfig)):
        return config_cls.from_config(raw)
    if isinstance(raw, Mapping):
        return config_cls.from_config(raw)
    raise TypeError(
        f"config must be {config_cls.__name__}, mapping, or None (got "
        f"{type(raw).__name__})."
    )


def load_taskset(
    taskset_id: str,
    config: TasksetConfig | Mapping[str, Any] | None = None,
) -> Taskset:
    """Resolve a taskset package by id and call its ``load_taskset(config)``.

    Mirrors :func:`verifiers.load_environment` for the v1 split.
    """
    logger.info(f"Loading taskset: {taskset_id}")
    module = import_taskset_module(taskset_id)
    config_cls = get_taskset_config_cls(module)
    cfg = _coerce_config(config, config_cls)
    taskset = module.load_taskset(cfg)
    if not isinstance(taskset, Taskset):
        raise TypeError(
            f"{module.__name__}.load_taskset must return a vf.Taskset "
            f"(got {type(taskset).__name__})."
        )
    logger.info(f"Successfully loaded taskset '{taskset_id}'")
    return taskset


def load_harness(
    harness_id: str,
    config: HarnessConfig | Mapping[str, Any] | None = None,
) -> Harness:
    """Resolve a harness package by id and call its ``load_harness(config)``."""
    logger.info(f"Loading harness: {harness_id}")
    module = resolve_harness_module(harness_id)
    config_cls = get_harness_config_cls(module)
    cfg = _coerce_config(config, config_cls)
    harness = module.load_harness(cfg)
    if not isinstance(harness, Harness):
        raise TypeError(
            f"{module.__name__}.load_harness must return a vf.Harness "
            f"(got {type(harness).__name__})."
        )
    logger.info(f"Successfully loaded harness '{harness_id}'")
    return harness
