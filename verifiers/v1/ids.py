"""The plugin / environment id, and on-demand installation from the Environments Hub.

A taskset, harness, or v0 environment is selected by an id in one of three forms:

  - ``name``              a local package (already importable / pip-installed)
  - ``org/name``          the env hub, latest version
  - ``org/name@version``  the env hub, a pinned version

`EnvId` is the type of every id field: a plain ``str`` validated against those forms (an
`Annotated[str, ...]`, not a subclass), so it flows unchanged through configs and the
wire. `env_name` derives the bare name (org / version stripped) for logging, display, and
output paths; `ensure_installed` makes a hub id importable on demand, reusing the same
install path as `prime env install`.
"""

import logging
from typing import Annotated

from pydantic import AfterValidator

from verifiers.utils.install_utils import (
    check_hub_env_installed,
    install_from_hub,
    is_hub_env,
    normalize_package_name,
    parse_env_id,
)

logger = logging.getLogger(__name__)


def _validate_env_id(env_id: str) -> str:
    """Validate the id's shape — a hub id must be a well-formed ``org/name[@version]``; a
    local id is any module name. Returns it unchanged (the value stays a plain ``str``)."""
    if is_hub_env(env_id):
        parse_env_id(env_id)  # raises ValueError on a malformed org/name[@version]
    return env_id


EnvId = Annotated[str, AfterValidator(_validate_env_id)]
"""A taskset / harness / environment id — ``name``, ``org/name``, or ``org/name@version``.
A plain validated ``str``; parse it with `env_name` / `env_module`."""


def env_name(env_id: str) -> str:
    """The bare name — org and version stripped (``org/gsm8k@1.0`` -> ``gsm8k``). Used for
    logging, display, and output paths."""
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
