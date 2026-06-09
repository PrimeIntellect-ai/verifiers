"""The plugin / environment id, and on-demand installation from the Environments Hub.

A taskset, harness, or v0 environment is selected by an id in one of three forms:

  - ``name``              a local package (already importable / pip-installed)
  - ``org/name``          the env hub, latest version
  - ``org/name@version``  the env hub, a pinned version

`EnvId` is the type of every such id field. It stays a plain string everywhere it is
serialized (CLI flag, config TOML, output path) and additionally parses into
``org`` / ``name`` / ``version`` — ``name`` being the normalized bare name used for
import, logging, and output paths. `ensure_installed` makes a hub id importable on demand,
reusing the same install path as `prime env install`.
"""

import logging

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from verifiers.utils.install_utils import (
    check_hub_env_installed,
    install_from_hub,
    is_hub_env,
    normalize_package_name,
    parse_env_id,
)

logger = logging.getLogger(__name__)


class EnvId(str):
    """A taskset / harness / environment id (``name``, ``org/name``, or ``org/name@version``).

    A `str` subclass, so it serializes as the original string and flows unchanged through
    configs and the wire; the parsed accessors (`org`, `name`, `version`, `module`) are
    derived on demand."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        # Validate as a str then wrap, so `--taskset.id org/name@v` and TOML `id = "..."`
        # parse to an EnvId and it dumps back to the same plain string.
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema()
        )

    @property
    def is_hub(self) -> bool:
        """Whether this id points at the env hub (``org/name`` form) vs a local package."""
        return is_hub_env(self)

    @property
    def org(self) -> str | None:
        """The hub owner (``org/name`` -> ``org``), or None for a local id."""
        return parse_env_id(self)[0] if self.is_hub else None

    @property
    def version(self) -> str | None:
        """The pinned version (``@version``), or None for a local id / hub latest."""
        return parse_env_id(self)[2] if self.is_hub else None

    @property
    def name(self) -> str:
        """The bare name, org and version stripped (``org/gsm8k@1.0`` -> ``gsm8k``) — used
        for logging, display, and output paths."""
        return parse_env_id(self)[1] if self.is_hub else str(self)

    @property
    def module(self) -> str:
        """The importable module name — `name` normalized (hyphens -> underscores)."""
        return normalize_package_name(self.name)


def ensure_installed(env_id: str) -> EnvId:
    """Make `env_id` importable, returning the parsed `EnvId` (import via its `module`).

    For a hub id (``org/name[@version]``) that isn't installed, install it from the
    Environments Hub — latest, or the pinned version — the same path `prime env install`
    uses. A local id is assumed already importable and returned unchanged."""
    eid = env_id if isinstance(env_id, EnvId) else EnvId(env_id)
    if eid.is_hub and not check_hub_env_installed(eid):
        logger.info("installing %s from the environments hub", eid)
        if not install_from_hub(eid):
            raise ModuleNotFoundError(
                f"could not install {eid!r} from the environments hub"
            )
    return eid
