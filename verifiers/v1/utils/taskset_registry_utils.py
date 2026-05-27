import sys
from typing import get_args, get_origin, get_type_hints

from ..config import TasksetConfig


TasksetType = type[object]


_TASKSET_CONFIG_TYPES: dict[TasksetType, type[TasksetConfig]] = {}


def register_taskset_config_type(
    taskset_type: TasksetType,
    config_type: type[TasksetConfig],
) -> None:
    existing = _TASKSET_CONFIG_TYPES.get(taskset_type)
    if existing is not None and existing is not config_type:
        raise TypeError(
            f"{taskset_type.__name__} is already registered to {existing.__name__}."
        )
    _TASKSET_CONFIG_TYPES[taskset_type] = config_type


def taskset_config_type(
    taskset_type: TasksetType,
    taskset_base: TasksetType,
) -> type[TasksetConfig]:
    for candidate in taskset_type.__mro__:
        config_type = _TASKSET_CONFIG_TYPES.get(candidate)
        if config_type is not None:
            return config_type
    return TasksetConfig


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
        if args:
            config_type = taskset_config_type_from_type_arg(args[0])
            if config_type is not None:
                return config_type
    return None


def taskset_config_type_from_type_arg(arg: object) -> type[TasksetConfig] | None:
    if isinstance(arg, type) and issubclass(arg, TasksetConfig):
        return arg
    bound = getattr(arg, "__bound__", None)
    if isinstance(bound, type) and issubclass(bound, TasksetConfig):
        return bound
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
