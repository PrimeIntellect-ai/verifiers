"""Environment/plugin id helpers + on-demand install from the Environments Hub.

Thin v1 wrappers over `verifiers.utils.install_utils`: derive an id's package/module name,
and make a hub id (`org/name[@version]`) importable on demand (the same path `prime env
install` uses)."""

import logging

from verifiers.utils.install_utils import (
    install_from_hub,
    is_hub_env,
    normalize_package_name,
    parse_env_id,
)

logger = logging.getLogger(__name__)
_installed_hub_refs: dict[str, str] = {}


def env_name(env_id: str) -> str:
    """The package name — the id with org and version stripped (``org/gsm8k@1.0`` ->
    ``gsm8k``). Used for logging, display, and output paths."""
    return parse_env_id(env_id)[1] if is_hub_env(env_id) else env_id


def env_module(env_id: str) -> str:
    """The importable module name — `env_name` normalized (hyphens -> underscores)."""
    return normalize_package_name(env_name(env_id))


def ensure_installed(env_id: str) -> str:
    """Make `env_id` importable and return its module name.

    A Hub reference is resolved once while it remains the active reference for its module.
    Switching between owners or versions of the same module reinstalls the requested package.
    A bare local id must already be importable (normally via ``uv pip install -e``)."""
    module = env_module(env_id)
    if is_hub_env(env_id):
        if _installed_hub_refs.get(module) != env_id:
            logger.info("installing %s from the environments hub", env_id)
            install_from_hub(env_id)
            _installed_hub_refs[module] = env_id
    return module
