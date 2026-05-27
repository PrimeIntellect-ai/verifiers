import sys
from typing import get_args, get_origin, get_type_hints

from ..config import HarnessConfig


HarnessType = type[object]


_HARNESS_CONFIG_TYPES: dict[HarnessType, type[HarnessConfig]] = {}


def register_harness_config_type(
    harness_type: HarnessType,
    config_type: type[HarnessConfig],
) -> None:
    existing = _HARNESS_CONFIG_TYPES.get(harness_type)
    if existing is not None and existing is not config_type:
        raise TypeError(
            f"{harness_type.__name__} is already registered to {existing.__name__}."
        )
    _HARNESS_CONFIG_TYPES[harness_type] = config_type


def harness_config_type(
    harness_type: HarnessType,
    harness_base: HarnessType,
) -> type[HarnessConfig]:
    for candidate in harness_type.__mro__:
        config_type = _HARNESS_CONFIG_TYPES.get(candidate)
        if config_type is not None:
            return config_type
    return HarnessConfig


def harness_config_type_from_class(
    harness_type: HarnessType,
    *,
    inherited: bool,
    harness_base: HarnessType,
) -> type[HarnessConfig] | None:
    bases = harness_type.__mro__ if inherited else (harness_type,)
    for base in bases:
        config_type = harness_config_type_from_orig_bases(base, harness_base)
        if config_type is not None:
            return config_type
        config_type = harness_config_type_from_annotation(base)
        if config_type is not None:
            return config_type
    return None


def harness_config_type_from_orig_bases(
    harness_type: HarnessType,
    harness_base: HarnessType,
) -> type[HarnessConfig] | None:
    for base in harness_type.__dict__.get("__orig_bases__", ()):
        origin = get_origin(base)
        if not isinstance(origin, type) or not issubclass(origin, harness_base):
            continue
        args = get_args(base)
        if args:
            config_type = harness_config_type_from_type_arg(args[0])
            if config_type is not None:
                return config_type
    return None


def harness_config_type_from_type_arg(arg: object) -> type[HarnessConfig] | None:
    if isinstance(arg, type) and issubclass(arg, HarnessConfig):
        return arg
    bound = getattr(arg, "__bound__", None)
    if isinstance(bound, type) and issubclass(bound, HarnessConfig):
        return bound
    return None


def harness_config_type_from_annotation(
    harness_type: HarnessType,
) -> type[HarnessConfig] | None:
    annotations = harness_type.__dict__.get("__annotations__", {})
    if "config" not in annotations:
        return None
    try:
        annotation = get_type_hints(harness_type).get("config")
    except Exception:
        annotation = resolve_harness_config_annotation(
            harness_type, annotations["config"]
        )
    if (
        isinstance(annotation, type)
        and issubclass(annotation, HarnessConfig)
        and annotation is not HarnessConfig
    ):
        return annotation
    return None


def resolve_harness_config_annotation(
    harness_type: HarnessType, annotation: object
) -> object:
    if not isinstance(annotation, str):
        return annotation
    module = sys.modules.get(harness_type.__module__)
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
