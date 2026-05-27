from collections.abc import Mapping
from typing import Literal, cast

from pydantic import field_validator

from .config import Config
from .types import ConfigData, ConfigInputMap
from .utils.config_utils import (
    explicit_config_data,
    resolved_config_data,
    string_mapping,
)


class SandboxConfig(Config):
    image: str = "python:3.11-slim"
    start_command: str = "tail -f /dev/null"
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    disk_size_gb: float = 5.0
    gpu_count: int = 0
    network_access: bool = True
    timeout_minutes: int = 60
    workdir: str | None = None
    command_timeout: int | None = None
    packages: list[str] = []
    install_timeout: int = 300
    setup_commands: list[str] = []
    setup_timeout: int = 300
    scope: Literal["rollout", "group", "global"] = "rollout"
    prefer: Literal["program"] | None = None

    @field_validator("packages", "setup_commands", mode="before")
    @classmethod
    def validate_string_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value


def sandbox_config_mapping(
    value: object | None, *, fill_defaults: bool = True
) -> ConfigData | None:
    if value is None:
        return None
    if isinstance(value, SandboxConfig):
        return (
            resolved_config_data(value)
            if fill_defaults
            else explicit_config_data(value, SandboxConfig)
        )
    if isinstance(value, Mapping):
        mapping = string_mapping(cast(ConfigInputMap, value))
        prefer = mapping.get("prefer")
        if prefer is not None and prefer != "program":
            raise ValueError("sandbox.prefer must be 'program'.")
        sandbox = SandboxConfig.model_validate(mapping).model_dump(exclude_none=True)
        if fill_defaults:
            return sandbox
        return {key: sandbox[key] for key in mapping if key in sandbox}
    raise TypeError("Sandbox config must be a mapping.")
