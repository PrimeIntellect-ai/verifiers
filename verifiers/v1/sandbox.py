from typing import Literal

from pydantic import Field, field_validator

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
    gpu_type: str | None = None
    vm: bool | None = None
    network_access: bool = True
    timeout_minutes: int = 60
    create_timeout: int | None = None
    wait_timeout: int | None = None
    environment_vars: dict[str, str] = {}
    secrets: dict[str, str] = {}
    team_id: str | None = None
    region: str | None = None
    registry_credentials_id: str | None = None
    guaranteed: bool = False
    workdir: str | None = None
    command_timeout: int | None = None
    poll_interval: int = 3
    packages: list[str] = []
    install_timeout: int = 300
    setup_commands: list[str] = []
    setup_timeout: int = 300
    labels: list[str] = []
    scope: Literal["rollout", "group", "global"] = "rollout"
    prefer: Literal["program"] | None = None
    create_concurrency: int = Field(default=128, ge=1)
    create_rate_per_second: float | None = Field(default=None, gt=0)
    delete_concurrency: int = Field(default=128, ge=1)
    delete_rate_per_second: float | None = Field(default=None, gt=0)

    @field_validator("packages", "setup_commands", "labels", mode="before")
    @classmethod
    def validate_string_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("environment_vars", "secrets", mode="before")
    @classmethod
    def validate_string_mapping(cls, value: object) -> object:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {str(key): str(item) for key, item in value.items()}
        return value

    def data(self, *, fill_defaults: bool = True) -> ConfigData:
        if fill_defaults:
            return resolved_config_data(self)
        return explicit_config_data(self, SandboxConfig)
