from collections.abc import Mapping
from typing import TypeAlias, cast

from pydantic import field_validator, model_validator

from .config import Config, ConfigSource
from .sandbox import SandboxConfig, sandbox_config_mapping
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
COMMAND_SANDBOX_DEFAULTS: ConfigData = {
    "image": "python:3.11-slim",
    "workdir": "/app",
    "scope": "rollout",
    "timeout_minutes": 120,
    "command_timeout": 900,
    "network_access": True,
}
COMMAND_PROGRAM_PATCH_KEYS = {
    "sandbox",
    "files",
    "dirs",
    "setup",
    "setup_timeout",
    "bindings",
    "env",
    "artifacts",
    "args",
}
COMMAND_PROGRAM_MAP_PATCH_KEYS = {"files", "dirs", "bindings", "env", "artifacts"}
COMMAND_PROGRAM_LIST_PATCH_KEYS = {"setup", "args"}

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
    sandbox: bool | SandboxConfig | ConfigData | None = None
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
        for name, value in (
            ("command", self.command),
            ("files", self.files),
            ("dirs", self.dirs),
            ("setup", self.setup),
            ("env", self.env),
            ("artifacts", self.artifacts),
            ("channels", self.channels),
            ("args", self.args),
        ):
            validate_program_value_refs(value, f"program.{name}")
        return self

    @classmethod
    def from_command(
        cls,
        *,
        command: ProgramCommand,
        program: "ProgramConfig | None" = None,
        sandbox: bool | ConfigData | SandboxConfig | None = None,
        default_sandbox: bool | ConfigData | SandboxConfig | None = True,
        sandbox_defaults: ConfigData | None = None,
        files: ProgramOptionMap | None = None,
        dirs: ProgramOptionMap | None = None,
        setup: ProgramSetup | None = None,
        setup_timeout: int | None = None,
        bindings: ConfigData | None = None,
        env: ProgramOptionMap | None = None,
        artifacts: ProgramOptionMap | None = None,
        channels: ProgramChannels | None = None,
        args: ProgramArgs | None = None,
    ) -> "ProgramConfig":
        patch = program or ProgramConfig()
        sandbox_value = (
            sandbox
            if sandbox is not None
            else patch.sandbox
            if patch.sandbox is not None
            else default_sandbox
            if default_sandbox is not None
            else True
        )
        data: ConfigData = {
            "command": command,
            "sandbox": command_sandbox_config(sandbox_value, defaults=sandbox_defaults)
            or False,
        }
        if files is not None:
            data["files"] = dict(files)
        if dirs is not None:
            data["dirs"] = dict(dirs)
        if setup is not None:
            data["setup"] = setup
        if setup_timeout is not None:
            data["setup_timeout"] = setup_timeout
        if bindings is not None:
            data["bindings"] = dict(bindings)
        if env is not None:
            data["env"] = dict(env)
        if artifacts is not None:
            data["artifacts"] = dict(artifacts)
        if channels is not None:
            data["channels"] = channels
        if args is not None:
            data["args"] = list(args)
        return cls.model_validate(merge_command_program_config(data, patch))


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


def command_sandbox_config(
    sandbox: bool | ConfigData | SandboxConfig,
    *,
    defaults: ConfigData | None = None,
) -> SandboxConfig | None:
    if sandbox is False:
        return None
    base = {**COMMAND_SANDBOX_DEFAULTS, **dict(defaults or {})}
    if sandbox is True:
        return SandboxConfig.model_validate(base)
    return SandboxConfig.model_validate(
        {**base, **(sandbox_config_mapping(sandbox) or {})}
    )


def merge_command_program_config(
    program: ConfigData,
    patch_config: ProgramConfig,
) -> ConfigData:
    patch = program_config_data(patch_config)
    unknown = sorted(set(patch) - COMMAND_PROGRAM_PATCH_KEYS)
    if unknown:
        allowed = ", ".join(sorted(COMMAND_PROGRAM_PATCH_KEYS))
        raise ValueError(
            f"Command ProgramConfig can only define {allowed}; got {unknown}."
        )
    merged: ConfigData = dict(program)
    for key, value in patch.items():
        if key in COMMAND_PROGRAM_MAP_PATCH_KEYS:
            if not isinstance(value, Mapping):
                raise TypeError(f"program.{key} must be a mapping.")
            base = merged.get(key, {})
            if base is None:
                base = {}
            if not isinstance(base, Mapping):
                raise TypeError(f"command program {key} must be a mapping.")
            merged[key] = {**dict(base), **dict(value)}
        elif key in COMMAND_PROGRAM_LIST_PATCH_KEYS:
            merged[key] = [
                *program_list_items(
                    cast(ProgramSetup | None, merged.get(key)),
                    f"command program {key}",
                ),
                *program_list_items(
                    cast(ProgramSetup | None, value),
                    f"program.{key}",
                ),
            ]
        else:
            merged[key] = value
    return merged


def program_list_items(
    value: ProgramSetup | None, field_name: str
) -> list[ProgramConfigValue]:
    if value is None:
        return []
    if isinstance(value, list):
        return cast(list[ProgramConfigValue], list(value))
    if isinstance(value, tuple):
        return cast(list[ProgramConfigValue], list(value))
    if isinstance(value, str) or isinstance(value, Mapping):
        return [cast(ProgramConfigValue, value)]
    raise TypeError(f"{field_name} must be a string, mapping, or list.")


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
