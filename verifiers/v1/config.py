from os import PathLike
from typing import Literal, TypeAlias

from pydantic import BaseModel, field_validator, model_validator
from pydantic_config import BaseConfig
from typing_extensions import Self

from .types import ConfigData
from .utils.config_utils import (
    annotation_text,
    coerce_config,
    default_text,
    import_config_ref as import_config_ref,
    resolve_config_object as resolve_config_object,
    string_mapping,
)


ConfigSource: TypeAlias = BaseModel | ConfigData | None


class Config(BaseConfig):
    """Strict serializable v1 config base."""

    @model_validator(mode="after")
    def validate_serializable_config(self) -> Self:
        for name in type(self).model_fields:
            try:
                validate_serializable_value(
                    self.__dict__[name], f"{type(self).__name__}.{name}"
                )
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
        return self

    @classmethod
    def schema_text(cls) -> str:
        lines = [cls.__name__]
        for name, field in cls.model_fields.items():
            lines.append(
                f"- {name}: {annotation_text(field.annotation)} = {default_text(field)}"
            )
        return "\n".join(lines)


def validate_serializable_value(value: object, field: str) -> None:
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, BaseModel):
        return
    if callable(value) or isinstance(value, PathLike):
        raise TypeError(f"{field} must be serializable; use an import ref string.")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} mapping keys must be strings.")
            validate_serializable_value(item, f"{field}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            validate_serializable_value(item, f"{field}.{index}")
        return
    raise TypeError(f"{field} must be serializable; got {type(value).__name__}.")


class CallableConfig(Config):
    fn: str
    priority: int | None = None
    stage: Literal["rollout", "group"] | None = None
    weight: float | None = None
    skip: bool = False


CallableEntry: TypeAlias = str | CallableConfig
ToolEntryData: TypeAlias = str | ConfigData
ToolsetData: TypeAlias = str | ConfigData
ToolsetCollectionData: TypeAlias = (
    ToolsetData | list[ToolsetData] | dict[str, ToolsetData]
)


class SignalConfig(Config):
    stage: Literal["rollout", "group"] | None = None
    priority: int | None = None
    weight: float | None = None
    skip: bool = False


def validate_scoring_map(value: object, field: str) -> dict[str, ConfigData]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a mapping.")
    result: dict[str, ConfigData] = {}
    for name, item in value.items():
        if not isinstance(name, str):
            raise TypeError(f"{field} keys must be strings.")
        if isinstance(item, BaseModel):
            data = item.model_dump(exclude_none=True, exclude_unset=True)
        elif isinstance(item, dict):
            data = string_mapping(item)
        else:
            raise TypeError(f"{field}.{name} must be a mapping.")
        result[name] = coerce_config(SignalConfig, data).model_dump(
            exclude_none=True,
            exclude_unset=True,
        )
    return result


class LifecycleConfig(Config):
    # Collection fields are configured only here; runtime mutation APIs are separate.
    toolsets: ToolsetCollectionData = []
    stops: list[CallableEntry] = []
    setups: list[CallableEntry] = []
    updates: list[CallableEntry] = []
    metrics: list[CallableEntry] = []
    rewards: list[CallableEntry] = []
    advantages: list[CallableEntry] = []
    cleanups: list[CallableEntry] = []
    teardowns: list[CallableEntry] = []
    scoring: dict[str, ConfigData] = {}

    @field_validator("scoring", mode="before")
    @classmethod
    def validate_scoring(cls, value: object) -> dict[str, ConfigData]:
        return validate_scoring_map(value, "scoring")
