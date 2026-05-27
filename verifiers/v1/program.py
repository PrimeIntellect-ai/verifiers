from collections.abc import Mapping
from typing import TypeAlias, cast

from pydantic import field_validator, model_validator

from .config import Config, ConfigSource, JsonMap
from .sandbox import SandboxConfig
from .types import (
    ConfigData,
    ConfigInputMap,
    ProgramArgs,
    ProgramChannels,
    ProgramCommand,
    ProgramConfigValue,
    ProgramOptionMap,
    ProgramSetup,
    ProgramValue as ProgramValue,
)
from .utils.binding_utils import Bindings, normalize_binding_map
from .utils.config_utils import coerce_config, explicit_config_data
from .utils.mcp_proxy_utils import validate_program_channels

ProgramCallableRef: TypeAlias = str

__all__ = [
    "Program",
    "ProgramArgs",
    "ProgramCallableRef",
    "ProgramChannels",
    "ProgramCommand",
    "ProgramConfig",
    "ProgramConfigValue",
    "ProgramOptionMap",
    "ProgramSetup",
    "ProgramValue",
    "program_config_data",
]


class ProgramConfig(Config):
    base: bool = False
    fn: ProgramCallableRef | None = None
    command: ProgramCommand | None = None
    sandbox: bool | SandboxConfig | JsonMap | None = None
    files: ProgramOptionMap = {}
    dirs: ProgramOptionMap = {}
    setup: ProgramSetup = []
    setup_timeout: int = 300
    bindings: Bindings = {}
    env: ProgramOptionMap = {}
    artifacts: ProgramOptionMap = {}
    channels: ProgramChannels | None = None
    args: ProgramArgs = []

    @field_validator("fn")
    @classmethod
    def validate_fn(cls, value: object) -> object:
        validate_program_callable_ref(value, "program.fn")
        return value

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: object) -> object:
        validate_program_channels(value)
        return value

    @field_validator("env", mode="before")
    @classmethod
    def validate_env(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
        return value

    @field_validator("env")
    @classmethod
    def normalize_env_values(cls, value: ProgramOptionMap) -> ProgramOptionMap:
        return {
            key: item if isinstance(item, Mapping) else str(item)
            for key, item in value.items()
        }

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> Bindings:
        return normalize_binding_map(value, "program.bindings", allow_objects=False)

    @model_validator(mode="after")
    def validate_program_callable_refs(self) -> "ProgramConfig":
        for name in (
            "command",
            "files",
            "dirs",
            "setup",
            "env",
            "artifacts",
            "channels",
            "args",
        ):
            validate_program_value_refs(getattr(self, name), f"program.{name}")
        return self


def validate_program_callable_ref(value: object, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty import ref string.")


def validate_program_value_refs(value: object, field_name: str) -> None:
    if isinstance(value, Mapping):
        mapping = cast(ConfigInputMap, value)
        if "fn" in mapping:
            validate_program_callable_ref(mapping["fn"], f"{field_name}.fn")
        for key, item in mapping.items():
            validate_program_value_refs(item, f"{field_name}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            validate_program_value_refs(item, f"{field_name}.{index}")


PROGRAM_DEFAULT_DUMP_DATA = ProgramConfig().model_dump(exclude_none=True)
PROGRAM_DEFAULT_DUMP_KEYS = set(PROGRAM_DEFAULT_DUMP_DATA)


def program_config_data(config: ProgramConfig) -> ConfigData:
    data = explicit_config_data(config)
    if PROGRAM_DEFAULT_DUMP_KEYS.issubset(config.model_fields_set):
        data = {
            key: value
            for key, value in data.items()
            if value != PROGRAM_DEFAULT_DUMP_DATA.get(key)
        }
    return data


class Program:
    config: ProgramConfig

    def __init__(self, config: ConfigSource = None):
        self.config = cast(ProgramConfig, coerce_config(ProgramConfig, config))

    def data(self) -> ConfigData:
        data = program_config_data(self.config)
        if data:
            return data
        return {
            "base": True,
        }
