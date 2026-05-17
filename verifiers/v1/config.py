from collections.abc import Iterable, Mapping
from pathlib import Path
import sys
from os import PathLike
from typing import Literal, TypeAlias, cast

from pydantic import BaseModel, ValidationInfo, field_validator, model_validator
from pydantic_config import BaseConfig
from typing_extensions import Self

from .types import ConfigData, ConfigInputMap, ConfigMap
from .utils.binding_utils import Bindings, normalize_binding_map
from .utils.config_callable_utils import (
    CallableKind as CallableKind,
    config_callables as config_callables,
    merge_config_handler_map as merge_config_handler_map,
)
from .utils.config_utils import (
    annotation_text,
    config_data,
    default_text,
    import_config_ref as import_config_ref,
    resolve_config_object as resolve_config_object,
    string_mapping,
)
from .utils.mcp_proxy_utils import validate_program_channels
from verifiers.types import ClientConfig

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


JsonMap: TypeAlias = ConfigData
ConfigSource: TypeAlias = BaseModel | ConfigMap | None
ProgramValue: TypeAlias = object
ProgramCommand: TypeAlias = str | list[ProgramValue]
ProgramOptionMap: TypeAlias = ConfigData
ProgramSetup: TypeAlias = ProgramValue | list[ProgramValue]
ProgramChannels: TypeAlias = str | JsonMap | list[str | JsonMap]
PromptInput: TypeAlias = str | list[JsonMap]
ToolsetSpecs: TypeAlias = str | JsonMap | list[str | JsonMap] | dict[str, str | JsonMap]
TaskSource: TypeAlias = str | list[JsonMap]


class Config(BaseConfig):
    """Strict serializable v1 config base."""

    @model_validator(mode="after")
    def validate_serializable_config(self) -> Self:
        for name in type(self).model_fields:
            validate_serializable_value(
                getattr(self, name), f"{type(self).__name__}.{name}"
            )
        return self

    @classmethod
    def from_config(cls, config: ConfigSource = None) -> Self:
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if isinstance(config, BaseModel):
            data = config.model_dump(exclude_none=True)
        elif isinstance(config, Mapping):
            data = string_mapping(cast(ConfigInputMap, config))
        else:
            raise TypeError("Config must be a mapping or config object.")
        return cls.model_validate(data)

    @classmethod
    def from_toml(
        cls, path: str | Path, section: str | Iterable[str] | None = None
    ) -> Self:
        with Path(path).open("rb") as f:
            data: object = tomllib.load(f)
        if section is not None:
            keys = section.split(".") if isinstance(section, str) else list(section)
            for key in keys:
                if not isinstance(data, Mapping):
                    raise TypeError(f"TOML section {section!r} does not exist.")
                data = data[key]
        return cls.from_config(cast(ConfigMap, data))

    @classmethod
    def schema_text(cls) -> str:
        lines = [cls.__name__]
        for name, field in cls.model_fields.items():
            lines.append(
                f"- {name}: {annotation_text(field.annotation)} = {default_text(field)}"
            )
        return "\n".join(lines)


def validate_serializable_value(value: object, field: str) -> None:
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, BaseModel):
        return
    if callable(value) or isinstance(value, PathLike):
        raise TypeError(f"{field} must be serializable; use an import ref string.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field} mapping keys must be strings.")
            validate_serializable_value(item, f"{field}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            validate_serializable_value(item, f"{field}.{index}")
        return
    raise TypeError(f"{field} must be serializable; got {type(value).__name__}.")


class CallableConfig(Config):
    fn: str
    priority: int | None = None
    stage: Literal["rollout", "group"] | None = None
    weight: float | None = None
    skip: bool = False


CallableEntry: TypeAlias = str | CallableConfig
CallableConfigEntry: TypeAlias = CallableEntry


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


class MCPToolConfig(Config):
    command: str
    args: list[str] = []
    env: dict[str, str] | None = None
    cwd: str | None = None

    @field_validator("args", mode="before")
    @classmethod
    def validate_args(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value


class ProgramConfig(Config):
    base: bool = False
    fn: str | None = None
    command: ProgramCommand | None = None
    sandbox: bool | SandboxConfig | JsonMap | None = None
    files: ProgramOptionMap = {}
    dirs: ProgramOptionMap = {}
    setup: ProgramSetup = []
    bindings: Bindings = {}
    env: ProgramOptionMap = {}
    artifacts: ProgramOptionMap = {}
    channels: ProgramChannels | None = None
    args: list[ProgramValue] = []

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: object) -> object:
        validate_program_channels(value)
        return value

    @field_validator("env", mode="before")
    @classmethod
    def validate_env(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): str(item) for key, item in value.items()}
        return value

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> Bindings:
        return normalize_binding_map(value, "program.bindings", allow_objects=False)


class UserConfig(Config):
    fn: str
    scope: Literal["rollout", "group", "global"] = "rollout"
    bindings: Bindings = {}
    objects: dict[str, str] = {}
    sandbox: SandboxConfig | None = None

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> Bindings:
        return normalize_binding_map(value, "user.bindings", key_style="arg")


class ToolsetConfig(Config):
    tools: str | JsonMap | list[str | JsonMap] | None = []
    show: list[str] | None = None
    hide: list[str] | None = None
    bindings: Bindings = {}
    objects: dict[str, str] = {}
    write: bool = False
    scope: Literal["rollout", "group", "global"] | None = None
    sandbox: SandboxConfig | Literal["program"] | None = None
    stops: list[CallableEntry] = []
    setups: list[CallableEntry] = []
    updates: list[CallableEntry] = []
    cleanups: list[CallableEntry] = []
    teardowns: list[CallableEntry] = []

    @field_validator("show", "hide", mode="before")
    @classmethod
    def validate_visibility_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> Bindings:
        return normalize_binding_map(value, "toolset.bindings")

    @model_validator(mode="after")
    def validate_visibility(self) -> "ToolsetConfig":
        if self.show is not None and self.hide is not None:
            raise ValueError("Toolset accepts show or hide, not both.")
        return self


class SignalConfig(Config):
    stage: Literal["rollout", "group"] | None = None
    priority: int | None = None
    weight: float | None = None
    skip: bool = False


def validate_scoring_map(value: object, field: str) -> dict[str, ConfigData]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping.")
    result: dict[str, ConfigData] = {}
    for name, item in value.items():
        if not isinstance(name, str):
            raise TypeError(f"{field} keys must be strings.")
        if isinstance(item, BaseModel):
            data = item.model_dump(exclude_none=True, exclude_unset=True)
        elif isinstance(item, Mapping):
            data = string_mapping(cast(ConfigInputMap, item))
        else:
            raise TypeError(f"{field}.{name} must be a mapping.")
        result[name] = SignalConfig.from_config(data).model_dump(
            exclude_none=True,
            exclude_unset=True,
        )
    return result


class TasksetConfig(Config):
    # Singleton fields describe one logical value owned by the taskset.
    source: TaskSource | None = None
    eval_source: TaskSource | None = None
    taskset_id: str | None = None
    system_prompt: PromptInput | None = None
    user: UserConfig | str | None = None
    bindings: Bindings = {}
    objects: dict[str, str] = {}

    # Collection fields are configured only here; runtime mutation APIs are separate.
    toolsets: ToolsetSpecs | None = []
    stops: list[CallableEntry] = []
    setups: list[CallableEntry] = []
    updates: list[CallableEntry] = []
    metrics: list[CallableEntry] = []
    rewards: list[CallableEntry] = []
    advantages: list[CallableEntry] = []
    cleanups: list[CallableEntry] = []
    scoring: dict[str, ConfigData] = {}

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> Bindings:
        return normalize_binding_map(value, "taskset.bindings")

    @field_validator("scoring", mode="before")
    @classmethod
    def validate_scoring(cls, value: object) -> dict[str, ConfigData]:
        return validate_scoring_map(value, "taskset.scoring")


class HarnessConfig(Config):
    # Singleton fields describe one logical value owned by the harness.
    program: ProgramConfig | str | None = None
    system_prompt: PromptInput | None = None
    system_prompt_merge: str = "reject"
    sandbox: SandboxConfig | None = None
    client: ClientConfig | JsonMap | str | None = None
    model: str | None = None
    sampling_args: JsonMap = {}
    keep_trajectory_step: str | None = None
    user: UserConfig | str | None = None
    bindings: Bindings = {}
    max_turns: int = 10

    # Collection fields are configured only here; runtime mutation APIs are separate.
    toolsets: ToolsetSpecs | None = []
    stops: list[CallableEntry] = []
    setups: list[CallableEntry] = []
    updates: list[CallableEntry] = []
    metrics: list[CallableEntry] = []
    rewards: list[CallableEntry] = []
    advantages: list[CallableEntry] = []
    cleanups: list[CallableEntry] = []
    scoring: dict[str, ConfigData] = {}

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> Bindings:
        return normalize_binding_map(value, "harness.bindings", allow_objects=False)

    @field_validator("scoring", mode="before")
    @classmethod
    def validate_scoring(cls, value: object) -> dict[str, ConfigData]:
        return validate_scoring_map(value, "harness.scoring")


class EnvConfig(Config):
    taskset: TasksetConfig = TasksetConfig()
    harness: HarnessConfig = HarnessConfig()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        extra_fields = set(cls.model_fields) - set(EnvConfig.model_fields)
        if extra_fields:
            raise TypeError(
                f"{cls.__name__} defines unsupported root env config fields: "
                f"{', '.join(sorted(extra_fields))}. Put env-specific settings on "
                "a TasksetConfig or HarnessConfig instead."
            )
        for field_name, expected_type in (
            ("taskset", TasksetConfig),
            ("harness", HarnessConfig),
        ):
            annotation = cls.model_fields[field_name].annotation
            if not (
                isinstance(annotation, type) and issubclass(annotation, expected_type)
            ):
                raise TypeError(
                    f"{cls.__name__}.{field_name} must be typed as a "
                    f"{expected_type.__name__} subclass."
                )

    @field_validator("taskset", "harness", mode="before")
    @classmethod
    def validate_child_config(cls, value: object, info: ValidationInfo) -> object:
        if value is None:
            raise ValueError(
                f"EnvConfig.{info.field_name} cannot be None. "
                "Omit the section to use the default config."
            )
        try:
            config_data(value)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        return value


def config_model_mapping(value: object | None) -> ConfigData | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return string_mapping(cast(ConfigInputMap, value))
    raise TypeError("Config value must be a mapping or config object.")


def sandbox_config_mapping(
    value: object | None, *, fill_defaults: bool = True
) -> ConfigData | None:
    if value is None:
        return None
    if isinstance(value, SandboxConfig):
        return value.model_dump(exclude_none=True, exclude_unset=not fill_defaults)
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
