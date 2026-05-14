import importlib
import inspect
from collections.abc import Mapping
from typing import cast

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from ..types import ConfigData, ConfigFactory, ConfigInputMap, ConfigMap


CONFIG_REF_COLLECTION_FIELDS = {
    "stops",
    "setups",
    "updates",
    "metrics",
    "rewards",
    "advantages",
    "cleanups",
    "teardowns",
}


def config_data(value: object, target: type[BaseModel] | None = None) -> ConfigData:
    if value is None:
        data: ConfigData = {}
    elif isinstance(value, BaseModel):
        data = model_config_data(value)
        if target is not None:
            data = {
                key: item for key, item in data.items() if key in target.model_fields
            }
    elif isinstance(value, Mapping):
        data = string_mapping(cast(ConfigInputMap, value))
    else:
        raise TypeError("Config must be a mapping or config object.")
    return data


def model_config_data(value: BaseModel) -> ConfigData:
    data: ConfigData = {}
    for key in value.model_fields_set:
        item = getattr(value, key)
        if item is not None:
            data[key] = config_dump_value(item)
    return data


def config_dump_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return model_config_data(value)
    if isinstance(value, Mapping):
        return {
            key: config_dump_value(item)
            for key, item in string_mapping(cast(ConfigInputMap, value)).items()
            if item is not None
        }
    if isinstance(value, list | tuple):
        return [config_dump_value(item) for item in value]
    return value


def omit_none(data: ConfigMap) -> ConfigData:
    return {key: value for key, value in data.items() if value is not None}


def merge_child_config(default: object, override: object) -> object:
    merged = deep_merge(config_data(default), config_data(override))
    if isinstance(default, BaseModel):
        return cast(type[BaseModel], type(default)).model_validate(merged)
    return merged


def expand_config_ref(value: object | None, target: type[BaseModel]) -> object | None:
    if not isinstance(value, Mapping):
        return value
    data = string_mapping(cast(ConfigInputMap, value))
    config_ref = data.pop("config", None)
    if config_ref is None:
        return data
    base = load_config_ref(config_ref)
    base_data = config_data(base, target)
    return merge_config_ref_overlay(base_data, data)


def expand_config_ref_data(data: ConfigData, target: type[BaseModel]) -> ConfigData:
    expanded = expand_config_ref(data, target)
    if not isinstance(expanded, dict):
        raise TypeError("config data must resolve to a mapping.")
    return cast(ConfigData, expanded)


def load_config_ref(config_ref: object) -> object:
    value = resolve_config_object(config_ref)
    if callable(value):
        value = cast(ConfigFactory, value)()
    if inspect.isawaitable(value):
        raise TypeError("config refs must resolve synchronously.")
    config_data(value)
    return value


def merge_config_ref_overlay(base: ConfigData, overlay: ConfigMap) -> ConfigData:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if (
            key in CONFIG_REF_COLLECTION_FIELDS
            and isinstance(existing, list)
            and isinstance(value, list)
        ):
            merged[key] = [*existing, *value]
        elif isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(
                string_mapping(cast(ConfigInputMap, existing)),
                string_mapping(cast(ConfigInputMap, value)),
            )
        else:
            merged[key] = value
    return merged


def merge_config_value(value: object, config: object) -> object:
    if config is None:
        return value
    if value is None:
        return config
    value_mapping = config_mapping(value)
    config_mapping_value = config_mapping(config)
    if value_mapping is not None and config_mapping_value is not None:
        return deep_merge(
            config_mapping_value,
            value_mapping,
        )
    return value


def config_mapping(value: object) -> ConfigData | None:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return string_mapping(cast(ConfigInputMap, value))
    return None


def resolve_config_object(value: object) -> object:
    if isinstance(value, str):
        return import_config_ref(value)
    return value


def import_config_ref(ref: str) -> object:
    module_name, separator, attr_path = ref.partition(":")
    if not separator or not module_name or not attr_path:
        raise ValueError(f"Config ref {ref!r} must use 'module:object'.")
    obj: object = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def deep_merge(base: ConfigData, overlay: ConfigMap) -> ConfigData:
    merged: ConfigData = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(
                string_mapping(cast(ConfigInputMap, existing)),
                string_mapping(cast(ConfigInputMap, value)),
            )
        else:
            merged[key] = value
    return merged


def string_mapping(value: ConfigInputMap) -> ConfigData:
    result: ConfigData = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Config mappings require string keys.")
        result[key] = item
    return result


def annotation_text(annotation: object) -> str:
    if getattr(annotation, "__args__", None):
        return str(annotation).replace("typing.", "")
    name = getattr(annotation, "__name__", None)
    if isinstance(name, str):
        return name
    return str(annotation).replace("typing.", "")


def default_text(field: object) -> str:
    default_factory = getattr(field, "default_factory", None)
    if default_factory is not None:
        return "<factory>"
    default = getattr(field, "default", PydanticUndefined)
    if default is PydanticUndefined:
        return "required"
    return repr(default)
