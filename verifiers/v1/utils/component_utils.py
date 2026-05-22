import importlib
import inspect
import types as py_types
from typing import Union, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from ..types import ConfigData, Handler


def component_id_from_data(
    data: ConfigData, *, alias_field: str, label: str
) -> str | None:
    id_value = data.get("id")
    alias_value = data.get(alias_field)
    if id_value is not None and not isinstance(id_value, str):
        raise TypeError(f"{label}.id must be a string.")
    if alias_value is not None and not isinstance(alias_value, str):
        raise TypeError(f"{label}.{alias_field} must be a string.")
    if id_value == "":
        raise ValueError(f"{label}.id must be a non-empty string.")
    if alias_value == "":
        raise ValueError(f"{label}.{alias_field} must be a non-empty string.")
    if id_value is not None and alias_value is not None and id_value != alias_value:
        raise ValueError(
            f"{label}.id and {label}.{alias_field} must match when both are provided."
        )
    return id_value if id_value is not None else alias_value


def component_config_data(
    *,
    data: ConfigData,
    component_id: str,
    alias_field: str,
    config_cls: type[BaseModel],
) -> ConfigData:
    config_data = dict(data)
    config_data.pop("id", None)
    config_data.pop(alias_field, None)
    if alias_field in config_cls.model_fields:
        config_data[alias_field] = component_id
    return config_data


def import_component_module(component_id: str, label: str) -> object:
    module_name = component_id.replace("-", "_").split("/")[-1]
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"Could not import {label} package {component_id!r}. "
            f"Ensure the '{component_id}' package is installed."
        ) from exc


def component_loader(
    module: object, loader_name: str, component_id: str, label: str
) -> Handler:
    if not hasattr(module, loader_name):
        module_name = component_id.replace("-", "_").split("/")[-1]
        raise AttributeError(
            f"Module '{module_name}' does not have a '{loader_name}' function. "
            f"Install the correct {label} package or add '{loader_name}' to it."
        )
    loader = getattr(module, loader_name)
    if not callable(loader):
        raise TypeError(f"{loader_name} on {component_id!r} must be callable.")
    return cast(Handler, loader)


def component_config_type(
    *,
    loader: Handler,
    loader_name: str,
    component_id: str,
    base_config_cls: type[BaseModel],
    label: str,
) -> type[BaseModel]:
    sig = inspect.signature(loader)
    param = sig.parameters.get("config")
    if param is None:
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} must define a "
            f"required 'config' parameter annotated as a {base_config_cls.__name__} subtype."
        )
    if param.default is not inspect.Parameter.empty:
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} must require "
            "'config' rather than providing a default."
        )
    if param.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} must accept "
            "'config' as a normal or keyword-only parameter."
        )
    for name in sig.parameters:
        if name == "config":
            continue
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} must only "
            "define a 'config' parameter."
        )
    try:
        annotation = get_type_hints(loader).get("config")
    except Exception:
        annotation = param.annotation
    if not is_strict_component_config_type(annotation, base_config_cls):
        raise TypeError(
            f"{loader_name} for {label} package {component_id!r} must annotate "
            f"'config' as a {base_config_cls.__name__} subtype."
        )
    return cast(type[BaseModel], annotation)


def call_component_loader(loader: Handler, *, config: BaseModel) -> object:
    return loader(config=config)


def is_strict_component_config_type(
    annotation: object, base_config_cls: type[BaseModel]
) -> bool:
    annotation_name = getattr(annotation, "__name__", "")
    if (
        annotation is inspect.Parameter.empty
        or annotation is object
        or annotation_name == "".join(("A", "n", "y"))
    ):
        return False
    origin = get_origin(annotation)
    if origin in (Union, py_types.UnionType) or get_args(annotation):
        return False
    return isinstance(annotation, type) and issubclass(annotation, base_config_cls)
