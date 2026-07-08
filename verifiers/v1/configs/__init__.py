"""Config schemas for the v1 entrypoints — the objects the CLIs parse."""

from typing import Any, TypeVar

from pydantic_config import BaseConfig
from pydantic_config import cli as _cli

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.configs.debug import DebugConfig
from verifiers.v1.configs.init import InitConfig
from verifiers.v1.configs.serve import ServeConfig
from verifiers.v1.configs.validate import ValidateConfig

T = TypeVar("T", bound=BaseConfig)


def cli(cls: type[T], **kwargs: Any) -> T:
    """`pydantic_config.cli` with `VF_*` env-var overrides enabled.

    Any config field is settable as `VF_<PATH>`, nesting levels joined with a
    double underscore (`VF_MODEL__NAME` sets `model.name`); TOML and CLI values
    for the same field win. Namespace constraint: `VF_CONFIG`, `VF_STATE_URL`,
    `VF_STATE_SECRET`, `VF_LOG_LEVEL` and `VF_BUILD_INPUTS` are process-control
    vars — a top-level config field with one of those names would collide.
    """
    kwargs.setdefault("env_prefix", "VF")
    return _cli(cls, **kwargs)


__all__ = [
    "DebugConfig",
    "EvalConfig",
    "InitConfig",
    "ServeConfig",
    "ValidateConfig",
    "cli",
]
