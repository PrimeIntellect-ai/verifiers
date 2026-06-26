"""Environment/plugin id helpers + on-demand install from the Environments Hub.

Thin v1 wrappers over `verifiers.utils.install_utils`: derive an id's package/module name,
and make a hub id (`org/name[@version]`) importable on demand (the same path `prime env
install` uses)."""

import logging
from functools import lru_cache

from verifiers.utils.install_utils import (
    install_from_hub,
    is_hub_env,
    normalize_package_name,
    parse_env_id,
)

logger = logging.getLogger(__name__)


def env_name(env_id: str) -> str:
    """The package name — the id with org and version stripped (``org/gsm8k@1.0`` ->
    ``gsm8k``). Used for logging, display, and output paths."""
    return parse_env_id(env_id)[1] if is_hub_env(env_id) else env_id


def env_module(env_id: str) -> str:
    """The importable module name — `env_name` normalized (hyphens -> underscores)."""
    return normalize_package_name(env_name(env_id))


@lru_cache(maxsize=None)
def ensure_installed(env_id: str) -> str:
    """Make `env_id` importable and return its module name.

    A Hub reference is resolved once per process so ``latest`` cannot silently use a stale
    install and two owners with the same distribution name cannot be confused. A bare local
    id must already be importable (normally via ``uv pip install -e``)."""
    if is_hub_env(env_id):
        logger.info("installing %s from the environments hub", env_id)
        install_from_hub(env_id)
    return env_module(env_id)
