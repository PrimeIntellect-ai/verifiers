from __future__ import annotations

from collections.abc import Mapping


def artifact_path(spec: Mapping[str, object]) -> str:
    path = spec.get("path")
    if not isinstance(path, str):
        raise TypeError("program artifact path must be a string.")
    return path


def artifact_format(spec: Mapping[str, object]) -> str:
    value = spec.get("format", "text")
    if not isinstance(value, str):
        raise TypeError("program artifact format must be a string.")
    return value


def artifact_key(spec: Mapping[str, object]) -> str | None:
    value = spec.get("key")
    if value is not None and not isinstance(value, str):
        raise TypeError("program artifact key must be a string.")
    return value


def artifact_optional(spec: Mapping[str, object]) -> bool:
    value = spec.get("optional", False)
    if not isinstance(value, bool):
        raise TypeError("program artifact optional must be a boolean.")
    return value
