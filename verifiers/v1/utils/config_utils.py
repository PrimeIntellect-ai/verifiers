import importlib
from collections.abc import Mapping
from typing import ClassVar, Generic, TypeVar, cast, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from ..types import ConfigData, ConfigInputMap

ConfigT = TypeVar("ConfigT", bound=BaseModel)
ConfigOwnerT = TypeVar("ConfigOwnerT", bound="ConfigBound")

_CONFIG_OWNERS: dict[tuple[type[BaseModel], type[BaseModel]], type["ConfigBound"]] = {}
_CONFIG_TYPE_ALIASES: dict[tuple[type[BaseModel], str], type[BaseModel]] = {}


class ConfigBound(Generic[ConfigT]):
    _config_base_cls: ClassVar[type[BaseModel]] = BaseModel
    _config_cls: ClassVar[type[BaseModel]] = BaseModel
    _config_aliases: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        explicit_config_cls = None
        for base in cls.__dict__.get("__orig_bases__", ()):
            origin = get_origin(base)
            if isinstance(origin, type) and issubclass(origin, ConfigBound):
                candidate = get_args(base)[0]
                if not isinstance(candidate, TypeVar):
                    explicit_config_cls = candidate
                break
        config_cls = explicit_config_cls or cls._config_cls
        if not isinstance(config_cls, type) or not issubclass(
            config_cls, cls._config_base_cls
        ):
            raise TypeError(
                f"{cls.__name__} generic argument must be "
                f"{cls._config_base_cls.__name__}."
            )
        cls._config_cls = config_cls
        if explicit_config_cls is not None and config_cls is not cls._config_base_cls:
            register_config_owner(
                base_cls=cls._config_base_cls,
                config_cls=config_cls,
                owner_cls=cls,
            )

    @classmethod
    def config_schema(cls) -> str:
        return cls._config_cls.schema_text()  # type: ignore[attr-defined]


def register_config_owner(
    *,
    base_cls: type[BaseModel],
    config_cls: type[BaseModel],
    owner_cls: type[ConfigOwnerT],
) -> None:
    key = (base_cls, config_cls)
    existing = _CONFIG_OWNERS.get(key)
    if existing is not None and existing is not owner_cls:
        raise TypeError(
            f"{config_cls.__name__} is already bound to {existing.__name__}; "
            f"define a distinct config class for {owner_cls.__name__}."
        )
    _CONFIG_OWNERS[key] = owner_cls
    for alias in (owner_cls.__name__, config_cls.__name__.removesuffix("Config")):
        key = "".join(char for char in alias.lower() if char.isalnum())
        _CONFIG_TYPE_ALIASES.setdefault((base_cls, key), config_cls)
    for alias in (
        *getattr(owner_cls, "_config_aliases", ()),
        *getattr(config_cls, "_config_aliases", ()),
    ):
        key = "".join(char for char in alias.lower() if char.isalnum())
        existing = _CONFIG_TYPE_ALIASES.get((base_cls, key))
        if existing is not None and existing is not config_cls:
            raise TypeError(
                f"Config alias {alias!r} already resolves to {existing.__name__}."
            )
        _CONFIG_TYPE_ALIASES[(base_cls, key)] = config_cls


def config_owner(
    config_cls: type[BaseModel],
    base_cls: type[BaseModel],
) -> type[ConfigBound] | None:
    for candidate in config_cls.__mro__:
        if not issubclass(candidate, BaseModel):
            continue
        owner = _CONFIG_OWNERS.get((base_cls, candidate))
        if owner is not None:
            return owner
    return None


def config_type_alias(
    alias: str,
    base_cls: type[BaseModel],
) -> type[BaseModel]:
    key = (base_cls, "".join(char for char in alias.lower() if char.isalnum()))
    config_cls = _CONFIG_TYPE_ALIASES.get(key)
    if config_cls is None:
        raise ValueError(f"Unknown config type {alias!r}.")
    return config_cls


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
