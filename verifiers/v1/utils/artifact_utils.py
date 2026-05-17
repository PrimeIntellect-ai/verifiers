from ..types import ConfigMap


def artifact_path(spec: ConfigMap) -> str:
    path = spec.get("path")
    if not isinstance(path, str):
        raise TypeError("program artifact path must be a string.")
    return path


def artifact_format(spec: ConfigMap) -> str:
    value = spec.get("format", "text")
    if not isinstance(value, str):
        raise TypeError("program artifact format must be a string.")
    return value


def artifact_key(spec: ConfigMap) -> str | None:
    value = spec.get("key")
    if value is not None and not isinstance(value, str):
        raise TypeError("program artifact key must be a string.")
    return value


def artifact_optional(spec: ConfigMap) -> bool:
    value = spec.get("optional", False)
    if not isinstance(value, bool):
        raise TypeError("program artifact optional must be a boolean.")
    return value
