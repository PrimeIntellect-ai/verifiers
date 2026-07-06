"""Environment/plugin id helpers.

Derive an id's package/module name. A hub id is `org/name[@version]`; a local id is a
plain module name. v1 assumes the package is already importable — the platform installs
environments (e.g. via `prime env install`); v1 never installs on demand."""

from verifiers.utils.install_utils import (
    is_hub_env,
    normalize_package_name,
    parse_env_id,
)


def env_name(env_id: str) -> str:
    """The package name — the id with org and version stripped (``org/gsm8k@1.0`` ->
    ``gsm8k``). Used for logging, display, and output paths."""
    return parse_env_id(env_id)[1] if is_hub_env(env_id) else env_id


def env_module(env_id: str) -> str:
    """The importable module name — `env_name` normalized (hyphens -> underscores)."""
    return normalize_package_name(env_name(env_id))
