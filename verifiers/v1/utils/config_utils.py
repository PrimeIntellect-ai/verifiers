import importlib
from collections.abc import Mapping
from typing import cast

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from ..types import ConfigData, ConfigInputMap, ConfigMap


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
