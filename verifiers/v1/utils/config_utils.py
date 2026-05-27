import importlib
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar, cast

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from ..types import ConfigData, ConfigInputMap

ConfigT = TypeVar("ConfigT", bound=BaseModel)
_CONFIG_REF_MODULE: ContextVar[str | None] = ContextVar(
    "CONFIG_REF_MODULE", default=None
)


def explicit_config_data(
    value: object, target: type[BaseModel] | None = None
) -> ConfigData:
    if value is None:
        data: ConfigData = {}
    elif isinstance(value, BaseModel):
        data = explicit_model_config_data(value)
        if target is not None:
            data = {
                key: item for key, item in data.items() if key in target.model_fields
            }
    elif isinstance(value, Mapping):
        data = string_mapping(cast(ConfigInputMap, value))
    else:
        raise TypeError("Config must be a mapping or config object.")
    return data


def coerce_config(config_cls: type[ConfigT], value: object = None) -> ConfigT:
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    return config_cls.model_validate(explicit_config_data(value))


def resolved_config_data(
    value: object, target: type[BaseModel] | None = None
) -> ConfigData:
    if value is None:
        data: ConfigData = {}
    elif isinstance(value, BaseModel):
        data = cast(ConfigData, value.model_dump(exclude_none=True))
        if target is not None:
            data = {
                key: item for key, item in data.items() if key in target.model_fields
            }
    elif isinstance(value, Mapping):
        data = string_mapping(cast(ConfigInputMap, value))
    else:
        raise TypeError("Config must be a mapping or config object.")
    return data


def explicit_model_config_data(value: BaseModel) -> ConfigData:
    data: ConfigData = {}
    for key in value.model_fields_set:
        item = getattr(value, key)
        data[key] = config_dump_value(item)
    return data


def config_dump_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return explicit_model_config_data(value)
    if isinstance(value, Mapping):
        return {
            key: config_dump_value(item)
            for key, item in string_mapping(cast(ConfigInputMap, value)).items()
        }
    if isinstance(value, list | tuple):
        return [config_dump_value(item) for item in value]
    return value


def resolve_config_object(value: object) -> object:
    if isinstance(value, str):
        return import_config_ref(value)
    return value


def current_config_ref_module() -> str | None:
    return _CONFIG_REF_MODULE.get()


def import_config_ref(ref: str) -> object:
    module_name, attr_path = config_ref_parts(ref)
    obj: object = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def qualified_config_ref(ref: str) -> str:
    module_name, attr_path = config_ref_parts(ref)
    return f"{module_name}:{attr_path}"


def config_ref_parts(ref: str) -> tuple[str, str]:
    module_name, separator, attr_path = ref.partition(":")
    if separator:
        if not module_name or not attr_path:
            raise ValueError(f"Config ref {ref!r} must use 'module:object'.")
    else:
        module_name = _CONFIG_REF_MODULE.get()
        attr_path = ref
        if module_name is None or not attr_path:
            raise ValueError(
                f"Config ref {ref!r} must use 'module:object' outside a config module."
            )
    return module_name, attr_path


@contextmanager
def config_ref_context(config: object) -> Iterator[None]:
    module_name = config_ref_module(config)
    if module_name is None:
        yield
        return
    token = _CONFIG_REF_MODULE.set(module_name)
    try:
        yield
    finally:
        _CONFIG_REF_MODULE.reset(token)


def config_ref_module(config: object) -> str | None:
    if isinstance(config, BaseModel):
        module_name = type(config).__module__
        if module_name != "verifiers.v1.config":
            return module_name
    return None


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
