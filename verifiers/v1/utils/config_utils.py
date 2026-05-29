import importlib
import sys
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from ..types import ConfigData, ConfigInputMap

ConfigT = TypeVar("ConfigT", bound=BaseModel)
ConfigOwner = type[object]
config_type_registry: dict[ConfigOwner, type[BaseModel]] = {}
FRAMEWORK_CONFIG_MODULES = {
    "verifiers.v1.config",
    "verifiers.v1.env",
    "verifiers.v1.harness",
    "verifiers.v1.model",
    "verifiers.v1.program",
    "verifiers.v1.sandbox",
    "verifiers.v1.taskset",
    "verifiers.v1.toolset",
    "verifiers.v1.user",
}
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


def register_config_type(
    owner_type: ConfigOwner,
    config_type: type[BaseModel],
) -> None:
    existing = config_type_registry.get(owner_type)
    if existing is not None and existing is not config_type:
        raise TypeError(
            f"{owner_type.__name__} is already registered to {existing.__name__}."
        )
    config_type_registry[owner_type] = config_type


def registered_config_type(
    owner_type: ConfigOwner,
    default_config_type: type[ConfigT],
) -> type[ConfigT]:
    for candidate in owner_type.__mro__:
        config_type = config_type_registry.get(candidate)
        if config_type is not None:
            return cast(type[ConfigT], config_type)
    return default_config_type


def config_type_from_class(
    owner_type: ConfigOwner,
    *,
    inherited: bool,
    owner_base: ConfigOwner,
    config_base: type[BaseModel],
) -> type[BaseModel] | None:
    bases = owner_type.__mro__ if inherited else (owner_type,)
    for base in bases:
        config_type = config_type_from_orig_bases(
            base, owner_base=owner_base, config_base=config_base
        )
        if config_type is not None:
            return config_type
        config_type = config_type_from_annotation(base, config_base=config_base)
        if config_type is not None:
            return config_type
    return None


def config_type_from_orig_bases(
    owner_type: ConfigOwner,
    *,
    owner_base: ConfigOwner,
    config_base: type[BaseModel],
) -> type[BaseModel] | None:
    for base in owner_type.__dict__.get("__orig_bases__", ()):
        origin = get_origin(base)
        if not isinstance(origin, type) or not issubclass(origin, owner_base):
            continue
        args = get_args(base)
        if args:
            config_type = config_type_from_type_arg(args[0], config_base)
            if config_type is not None:
                return config_type
    return None


def config_type_from_type_arg(
    arg: object,
    config_base: type[BaseModel],
) -> type[BaseModel] | None:
    if isinstance(arg, type) and issubclass(arg, config_base):
        return arg
    bound = getattr(arg, "__bound__", None)
    if isinstance(bound, type) and issubclass(bound, config_base):
        return bound
    return None


def config_type_from_annotation(
    owner_type: ConfigOwner,
    *,
    config_base: type[BaseModel],
) -> type[BaseModel] | None:
    annotations = owner_type.__dict__.get("__annotations__", {})
    if "config" not in annotations:
        return None
    try:
        annotation = get_type_hints(owner_type).get("config")
    except Exception:
        annotation = resolve_config_annotation(owner_type, annotations["config"])
    if (
        isinstance(annotation, type)
        and issubclass(annotation, config_base)
        and annotation is not config_base
    ):
        return annotation
    return None


def resolve_config_annotation(owner_type: ConfigOwner, annotation: object) -> object:
    if not isinstance(annotation, str):
        return annotation
    module = sys.modules.get(owner_type.__module__)
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
        if module_name not in FRAMEWORK_CONFIG_MODULES:
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
