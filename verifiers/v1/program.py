from typing import Literal, TypeAlias, cast

from pydantic import field_validator, model_validator
from typing_extensions import TypeAliasType

from .artifact import ArtifactsConfig
from .config import Config
from .sandbox import SandboxConfig
from .types import ConfigData, ConfigValue
from .utils.binding_utils import BindingsConfig
from .utils.config_utils import explicit_config_data, string_mapping
from .utils.mcp_proxy_utils import validate_program_channels

ProgramCallableRef: TypeAlias = str
ProgramScalar: TypeAlias = str | int | float | bool | None
ProgramValue = TypeAliasType(
    "ProgramValue",
    ProgramScalar | list["ProgramValue"] | dict[str, "ProgramValue"],
)
ProgramCommand: TypeAlias = str | list[ProgramValue]
ProgramFiles: TypeAlias = dict[str, ProgramValue]
ProgramDirs: TypeAlias = dict[str, ProgramValue]
ProgramEnv: TypeAlias = dict[str, ProgramValue]
ProgramArtifacts: TypeAlias = ArtifactsConfig
ProgramSetup: TypeAlias = ProgramValue | list[ProgramValue]
ProgramArgs: TypeAlias = list[ProgramValue]
ProgramChannel: TypeAlias = Literal["callable", "mcp"]
ProgramChannelConfig: TypeAlias = ProgramChannel | dict[str, ProgramValue]
ProgramChannels: TypeAlias = ProgramChannelConfig | list[ProgramChannelConfig]
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
    "ProgramArgs",
    "ProgramArtifacts",
    "ProgramCallableRef",
    "ProgramChannels",
    "ProgramCommand",
    "ProgramConfig",
    "ProgramDirs",
    "ProgramEnv",
    "ProgramFiles",
    "ProgramSetup",
    "ProgramValue",
    "program_config_data",
]


class ProgramConfig(Config):
    base: bool = False
    fn: ProgramCallableRef | None = None
    command: ProgramCommand | None = None
    sandbox: bool | SandboxConfig | None = None
    files: ProgramFiles = {}
    dirs: ProgramDirs = {}
    setup: ProgramSetup = []
    setup_timeout: int = 300
    bindings: BindingsConfig = BindingsConfig()
    env: ProgramEnv = {}
    artifacts: ArtifactsConfig = ArtifactsConfig()
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
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
        return value

    @field_validator("env")
    @classmethod
    def normalize_env_values(cls, value: ProgramEnv) -> ProgramEnv:
        return {
            key: item if isinstance(item, dict) else str(item)
            for key, item in value.items()
        }

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> BindingsConfig:
        bindings = BindingsConfig.model_validate(value or {})
        bindings.entries("program.bindings", allow_objects=False)
        return bindings

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

    def data(self) -> ConfigData:
        resolved = self.resolve()
        if resolved is not self:
            return resolved.data()
        data = program_config_data(self)
        return data if data else {"base": True}

    def resolve(self) -> "ProgramConfig":
        return self

    def resolve_command(
        self,
        *,
        command: ProgramCommand,
        sandbox: bool | SandboxConfig | None = None,
        default_sandbox: bool | SandboxConfig | None = True,
        sandbox_defaults: ConfigData | None = None,
        files: ProgramFiles | None = None,
        dirs: ProgramDirs | None = None,
        setup: ProgramSetup | None = None,
        setup_timeout: int | None = None,
        bindings: ConfigData | None = None,
        env: ProgramEnv | None = None,
        artifacts: ArtifactsConfig | ConfigData | None = None,
        channels: ProgramChannels | None = None,
        args: ProgramArgs | None = None,
    ) -> "ProgramConfig":
        sandbox_value = (
            sandbox
            if sandbox is not None
            else self.sandbox
            if self.sandbox is not None
            else default_sandbox
            if default_sandbox is not None
            else True
        )
        resolved_sandbox = command_sandbox_config(
            sandbox_value, defaults=sandbox_defaults
        )
        data: ConfigData = {
            "command": cast(ConfigValue, command),
            "sandbox": resolved_sandbox.model_dump(exclude_none=True)
            if resolved_sandbox is not None
            else False,
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
            data["artifacts"] = ArtifactsConfig.model_validate(artifacts).data(
                "program.artifacts"
            )
        if channels is not None:
            data["channels"] = channels
        if args is not None:
            data["args"] = list(args)
        return ProgramConfig.model_validate(merge_command_program_config(data, self))


def validate_program_callable_ref(value: object, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty import ref string.")


def validate_program_value_refs(value: object, field_name: str) -> None:
    if isinstance(value, dict):
        mapping = string_mapping(value)
        if "fn" in mapping:
            validate_program_callable_ref(mapping["fn"], f"{field_name}.fn")
        for key, item in mapping.items():
            validate_program_value_refs(item, f"{field_name}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            validate_program_value_refs(item, f"{field_name}.{index}")


def command_sandbox_config(
    sandbox: bool | SandboxConfig,
    *,
    defaults: ConfigData | None = None,
) -> SandboxConfig | None:
    if sandbox is False:
        return None
    base = {**COMMAND_SANDBOX_DEFAULTS, **dict(defaults or {})}
    if sandbox is True:
        return SandboxConfig.model_validate(base)
    return SandboxConfig.model_validate({**base, **sandbox.data()})


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
        if key == "sandbox" and "sandbox" in merged:
            continue
        if key in COMMAND_PROGRAM_MAP_PATCH_KEYS:
            if not isinstance(value, dict):
                raise TypeError(f"program.{key} must be a mapping.")
            base = merged.get(key, {})
            if base is None:
                base = {}
            if not isinstance(base, dict):
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
) -> list[ProgramValue]:
    if value is None:
        return []
    if isinstance(value, list):
        return cast(list[ProgramValue], list(value))
    if isinstance(value, str) or isinstance(value, dict):
        return [cast(ProgramValue, value)]
    raise TypeError(f"{field_name} must be a string, mapping, or list.")


PROGRAM_DEFAULT_DUMP_DATA = ProgramConfig().model_dump(exclude_none=True)
PROGRAM_DEFAULT_DUMP_KEYS = set(PROGRAM_DEFAULT_DUMP_DATA)


def program_config_data(config: ProgramConfig) -> ConfigData:
    data = {
        key: value
        for key, value in explicit_config_data(config).items()
        if key in ProgramConfig.model_fields
    }
    if PROGRAM_DEFAULT_DUMP_KEYS.issubset(config.model_fields_set):
        data = {
            key: value
            for key, value in data.items()
            if value != PROGRAM_DEFAULT_DUMP_DATA.get(key)
        }
    return data
