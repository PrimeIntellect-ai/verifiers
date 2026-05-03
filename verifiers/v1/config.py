from __future__ import annotations

import importlib
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import PydanticUndefined


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @classmethod
    def from_toml(
        cls, path: str | Path, section: str | Iterable[str] | None = None
    ) -> Self:
        with Path(path).open("rb") as f:
            data: object = tomllib.load(f)
        if section is not None:
            keys = section.split(".") if isinstance(section, str) else list(section)
            for key in keys:
                if not isinstance(data, Mapping):
                    raise TypeError(f"TOML section {section!r} does not exist.")
                data = data[key]
        return cls.model_validate(data)

    @classmethod
    def schema_text(cls) -> str:
        lines = [cls.__name__]
        for name, field in cls.model_fields.items():
            lines.append(
                f"- {name}: {annotation_text(field.annotation)} = {default_text(field)}"
            )
        return "\n".join(lines)


class TasksetConfig(Config):
    source: object | None = None
    eval_source: object | None = None
    taskset_id: str | None = None
    toolsets: list[object] = Field(default_factory=list)
    user: object | None = None
    metrics: list[object] = Field(default_factory=list)
    rewards: list[object] = Field(default_factory=list)
    cleanup: list[object] = Field(default_factory=list)
    scoring: dict[str, dict[str, object]] = Field(default_factory=dict)


class HarnessConfig(Config):
    program: object | None = None
    sandbox: object | None = None
    toolsets: list[object] = Field(default_factory=list)
    user: object | None = None
    metrics: list[object] = Field(default_factory=list)
    rewards: list[object] = Field(default_factory=list)
    cleanup: list[object] = Field(default_factory=list)
    scoring: dict[str, dict[str, object]] = Field(default_factory=dict)
    max_turns: int = 10


def merge_config_value(value: object, config: object) -> object:
    if config is None:
        return value
    if value is None:
        return config
    if isinstance(value, Mapping) and isinstance(config, Mapping):
        return deep_merge(
            string_mapping(cast(Mapping[object, object], config)),
            string_mapping(cast(Mapping[object, object], value)),
        )
    return value


def merge_config_items(values: Iterable[object], config: object) -> list[object]:
    return [*values, *config_items(config)]


def config_items(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [import_config_ref(value)]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Iterable):
        return [resolve_config_object(item) for item in value]
    return [value]


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


def deep_merge(
    base: dict[str, object], overlay: Mapping[str, object]
) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(
                string_mapping(cast(Mapping[object, object], existing)),
                string_mapping(cast(Mapping[object, object], value)),
            )
        else:
            merged[key] = value
    return merged


def string_mapping(value: Mapping[object, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Config mappings require string keys.")
        result[key] = item
    return result


def annotation_text(annotation: Any) -> str:
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
