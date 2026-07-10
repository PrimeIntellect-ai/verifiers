"""Environment/plugin ID helpers and on-demand Hub installation."""

import logging

from verifiers.utils.install_utils import (
    check_hub_env_installed,
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


def ensure_installed(env_id: str) -> str:
    """Make `env_id` importable and return its module name.

    For a hub id (``org/name[@version]``) that isn't installed, install it from the
    Environments Hub — latest, or the pinned version — the same path `prime env install`
    uses. A local id is assumed already importable."""
    if is_hub_env(env_id) and not check_hub_env_installed(env_id):
        logger.info("installing %s from the environments hub", env_id)
        if not install_from_hub(env_id):
            raise ModuleNotFoundError(
                f"could not install {env_id!r} from the environments hub"
            )
    return env_module(env_id)
