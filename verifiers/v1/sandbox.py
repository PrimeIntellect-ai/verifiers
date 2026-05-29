from typing import Literal

from pydantic import field_validator

from .config import Config
from .types import ConfigData
from .utils.config_utils import (
    explicit_config_data,
    resolved_config_data,
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

    def data(self, *, fill_defaults: bool = True) -> ConfigData:
        if fill_defaults:
            return resolved_config_data(self)
        return explicit_config_data(self, SandboxConfig)
