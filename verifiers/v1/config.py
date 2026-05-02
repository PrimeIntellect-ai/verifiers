from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pydantic import BaseModel, ConfigDict, Field


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class TasksetConfig(Config):
    source: object | None = None
    scoring: dict[str, dict[str, object]] = Field(default_factory=dict)
    toolsets: object | None = None
    cleanup: object | None = None


class HarnessConfig(Config):
    program: object | None = None
    sandbox: object | None = None
    toolsets: object | None = None
    scoring: dict[str, dict[str, object]] = Field(default_factory=dict)
    cleanup: object | None = None


def merge_config_value(value: object, config: object) -> object:
    if config is None:
        return value
    if value is None:
        return config
    if isinstance(value, Mapping) and isinstance(config, Mapping):
        return deep_merge(
            string_mapping(cast(Mapping[object, object], value)),
            string_mapping(cast(Mapping[object, object], config)),
        )
    return value


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
