import sys
from typing import get_args, get_origin, get_type_hints

from ..config import TasksetConfig


TasksetType = type[object]


_TASKSET_REGISTRY: dict[type[TasksetConfig], TasksetType] = {}


def register_taskset_type(
    config_type: type[TasksetConfig],
    taskset_type: TasksetType,
) -> None:
    existing = _TASKSET_REGISTRY.get(config_type)
    if existing is not None and existing is not taskset_type:
        raise TypeError(
            f"{config_type.__name__} is already registered to {existing.__name__}."
        )
    _TASKSET_REGISTRY[config_type] = taskset_type


def taskset_type_for_config(
    config_type: type[TasksetConfig],
) -> TasksetType | None:
    for candidate in config_type.__mro__:
        if not issubclass(candidate, TasksetConfig):
            continue
        taskset_type = _TASKSET_REGISTRY.get(candidate)
        if taskset_type is not None:
            return taskset_type
    return None


def taskset_config_type(
    taskset_type: TasksetType,
    taskset_base: TasksetType,
) -> type[TasksetConfig]:
    config_type = taskset_config_type_from_class(
        taskset_type, inherited=True, taskset_base=taskset_base
    )
    return config_type or TasksetConfig


def taskset_config_type_from_class(
    taskset_type: TasksetType,
    *,
    inherited: bool,
    taskset_base: TasksetType,
) -> type[TasksetConfig] | None:
    bases = taskset_type.__mro__ if inherited else (taskset_type,)
    for base in bases:
        config_type = taskset_config_type_from_orig_bases(base, taskset_base)
        if config_type is not None:
            return config_type
        config_type = taskset_config_type_from_annotation(base)
        if config_type is not None:
            return config_type
    return None


def taskset_config_type_from_orig_bases(
    taskset_type: TasksetType,
    taskset_base: TasksetType,
) -> type[TasksetConfig] | None:
    for base in taskset_type.__dict__.get("__orig_bases__", ()):
        origin = get_origin(base)
        if not isinstance(origin, type) or not issubclass(origin, taskset_base):
            continue
        args = get_args(base)
        if args and isinstance(args[0], type) and issubclass(args[0], TasksetConfig):
            return args[0]
    return None


def taskset_config_type_from_annotation(
    taskset_type: TasksetType,
) -> type[TasksetConfig] | None:
    annotations = taskset_type.__dict__.get("__annotations__", {})
    if "config" not in annotations:
        return None
    try:
        annotation = get_type_hints(taskset_type).get("config")
    except Exception:
        annotation = resolve_taskset_config_annotation(
            taskset_type, annotations["config"]
        )
    if (
        isinstance(annotation, type)
        and issubclass(annotation, TasksetConfig)
        and annotation is not TasksetConfig
    ):
        return annotation
    return None


def resolve_taskset_config_annotation(
    taskset_type: TasksetType, annotation: object
) -> object:
    if not isinstance(annotation, str):
        return annotation
    module = sys.modules.get(taskset_type.__module__)
    if module is None:
        return None
    value: object = module.__dict__
    for part in annotation.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            return None
    return value
