"""Shared v1 taskset/harness builders used by the ``vf-eval-v1`` CLI and by
``verifiers.utils.env_utils.load_environment`` when it has to materialize a v1
env from inside a worker subprocess.

Two private kwargs are reserved on ``vf.load_environment(env_id, **env_args)``:

* ``__vf_v1_taskset__`` — dict merged into the env's TasksetConfig subclass.
* ``__vf_v1_harness__`` — dict describing the harness selection + overrides.
  Recognised shape: ``{"name": "<alias-or-import-ref>"|None, **overrides}``.

The CLI emits those when the user customises a v1 env. Workers see the same
``env_args`` and re-run the same builder, so a configured taskset/harness pair
ends up identical in every process.
"""

import inspect
from typing import Any, cast, get_args, get_origin

import verifiers as vf
from verifiers.v1.config import HarnessConfig, TasksetConfig
from verifiers.v1.env import Env as VEnv
from verifiers.v1.harness import Harness
from verifiers.v1.taskset import Taskset
from verifiers.v1.utils.config_utils import import_config_ref

V1_TASKSET_KEY = "__vf_v1_taskset__"
V1_HARNESS_KEY = "__vf_v1_harness__"

# Short aliases for harnesses bundled under ``verifiers.v1.packages.harnesses``.
# Any ``pkg.mod:Class`` import ref is also accepted on ``--harness.name``.
HARNESS_ALIASES: dict[str, str] = {
    "base": "verifiers.v1:Harness",
    "rlm": "verifiers.v1.packages.harnesses:RLM",
    "opencode": "verifiers.v1.packages.harnesses:OpenCode",
    "pi": "verifiers.v1.packages.harnesses:Pi",
    "mini-swe-agent": "verifiers.v1.packages.harnesses:MiniSWEAgent",
    "terminus-2": "verifiers.v1.packages.harnesses:Terminus2",
}


def has_v1_overrides(env_args: dict[str, Any]) -> bool:
    """True if ``env_args`` carries the v1 dispatch markers."""
    return V1_TASKSET_KEY in env_args or V1_HARNESS_KEY in env_args


def module_supports_v1_loader(module) -> bool:
    return hasattr(module, "load_taskset")


def resolve_harness_class(name: str) -> type[Harness]:
    target = HARNESS_ALIASES.get(name, name)
    if ":" not in target:
        raise ValueError(
            f"harness name {name!r} must be a registry alias "
            f"({sorted(HARNESS_ALIASES)}) or a 'pkg.mod:Class' import ref."
        )
    obj = import_config_ref(target)
    if not (isinstance(obj, type) and issubclass(obj, Harness)):
        raise TypeError(
            f"harness name {name!r} resolved to {obj!r}, which is not a "
            f"verifiers.v1.Harness subclass."
        )
    return obj


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


def _factory_config_type(module, factory_name: str, base_type: type) -> type | None:
    # Lazy import to avoid pulling env_utils into module init.
    from verifiers.utils.env_utils import factory_config_type

    return factory_config_type(module, factory_name, base_type)


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
    spec = dict(harness_spec)
    name = spec.pop("name", None)
    if name:
        harness_cls = resolve_harness_class(name)
        config_type = harness_config_type_from_class(harness_cls)
        config = config_type.model_validate(spec)
        harness = harness_cls(config=config)
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
    from verifiers.utils.env_utils import import_env_module

    env_module = import_env_module(env_id)
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


def pop_v1_overrides(env_args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
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
    "HARNESS_ALIASES",
    "V1_HARNESS_KEY",
    "V1_TASKSET_KEY",
    "build_v1_env",
    "build_v1_harness",
    "build_v1_taskset",
    "harness_config_type_from_class",
    "has_v1_overrides",
    "module_supports_v1_loader",
    "pop_v1_overrides",
    "resolve_harness_class",
]
