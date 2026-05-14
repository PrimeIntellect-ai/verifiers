from collections.abc import Iterable, Mapping
from pathlib import Path
import sys
from typing import ClassVar, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticUndefined
from typing_extensions import Self

from .types import (
    CallableConfigEntry,
    ConfigData,
    ConfigInputMap,
    ConfigMap,
    ConfigSource,
    Handler,
    ModelClient,
    Objects,
    ProgramCommand,
    ProgramOptionMap,
    ProgramSetup,
    ProgramChannels,
    ProgramValue,
    PromptInput,
    TaskSource,
    ToolsetSpecs,
    ToolSpecs,
)
from .utils.binding_utils import (
    Bindings,
    normalize_binding_map,
    normalize_object_map,
)
from .utils.config_callable_utils import (
    CallableKind as CallableKind,
    config_callables as config_callables,
    merge_config_handler_map as merge_config_handler_map,
)
from .utils.config_utils import (
    annotation_text,
    config_data,
    default_text,
    expand_config_ref,
    expand_config_ref_data,
    import_config_ref as import_config_ref,
    merge_child_config,
    merge_config_value as merge_config_value,
    omit_none,
    resolve_config_object as resolve_config_object,
    string_mapping as string_mapping,
)
from .utils.mcp_proxy_utils import validate_program_channels

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    supports_config_ref: ClassVar[bool] = False

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        changed = False
        for field in cls.model_fields.values():
            annotation = field.annotation
            if (
                field.is_required()
                and isinstance(annotation, type)
                and issubclass(annotation, Config)
            ):
                field.default = PydanticUndefined
                field.default_factory = annotation
                changed = True
        if changed:
            cls.model_rebuild(force=True)

    def __init__(self, config: ConfigSource | None = None, /, **data: object):
        super().__init__(**type(self)._merge_config_data(config, data))

    @classmethod
    def from_config(cls, config: ConfigSource | None = None, /, **data: object) -> Self:
        return cls(**cls._merge_config_data(config, data))

    @classmethod
    def _merge_config_data(
        cls, config: ConfigSource | None, data: ConfigData
    ) -> ConfigData:
        data = omit_none(data)
        if cls.supports_config_ref:
            cls._validate_config_ref_contract()
            config = cast(ConfigSource | None, expand_config_ref(config, cls))
            data = expand_config_ref_data(data, cls)
        if config is not None:
            base = config_data(config, cls)
            base.update(data)
            data = base
        return data

    @classmethod
    def _validate_config_ref_contract(cls) -> None:
        if "config" in cls.model_fields:
            raise TypeError(
                f"{cls.__name__} reserves the 'config' field for config refs."
            )

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
        return cls.from_config(data)

    @classmethod
    def schema_text(cls) -> str:
        lines = [cls.__name__]
        for name, field in cls.model_fields.items():
            lines.append(
                f"- {name}: {annotation_text(field.annotation)} = {default_text(field)}"
            )
        return "\n".join(lines)


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
    packages: list[str] = Field(default_factory=list)
    install_timeout: int = 300
    setup_commands: list[str] = Field(default_factory=list)
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
    args: list[str] = Field(default_factory=list)
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
    sandbox: bool | SandboxConfig | ConfigMap | None = None
    files: ProgramOptionMap = Field(default_factory=dict)
    dirs: ProgramOptionMap = Field(default_factory=dict)
    setup: ProgramSetup = Field(default_factory=list)
    bindings: Bindings = Field(default_factory=dict)
    env: ProgramOptionMap = Field(default_factory=dict)
    artifacts: ProgramOptionMap = Field(default_factory=dict)
    channels: ProgramChannels | None = None
    args: list[ProgramValue] = Field(default_factory=list)

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: object) -> object:
        validate_program_channels(value)
        return value

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> Bindings:
        return normalize_binding_map(value, "program.bindings", allow_objects=False)


class UserConfig(Config):
    fn: Handler | str
    scope: Literal["rollout", "group", "global"] = "rollout"
    bindings: Bindings = Field(default_factory=dict)
    objects: Objects = Field(default_factory=dict)
    sandbox: SandboxConfig | None = None

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> Bindings:
        return normalize_binding_map(value, "user.bindings", key_style="arg")

    @field_validator("objects", mode="before")
    @classmethod
    def validate_objects(cls, value: object) -> Objects:
        return normalize_object_map(value, "user.objects")


class ToolsetConfig(Config):
    tools: ToolSpecs | None = Field(default_factory=list)
    show: list[str] | None = None
    hide: list[str] | None = None
    bindings: Bindings = Field(default_factory=dict)
    objects: Objects = Field(default_factory=dict)
    write: bool = False
    scope: Literal["rollout", "group", "global"] | None = None
    sandbox: SandboxConfig | Literal["program"] | None = None
    stops: list[CallableConfigEntry] = Field(default_factory=list)
    setups: list[CallableConfigEntry] = Field(default_factory=list)
    updates: list[CallableConfigEntry] = Field(default_factory=list)
    cleanups: list[CallableConfigEntry] = Field(default_factory=list)
    teardowns: list[CallableConfigEntry] = Field(default_factory=list)

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

    @field_validator("objects", mode="before")
    @classmethod
    def validate_objects(cls, value: object) -> Objects:
        return normalize_object_map(value, "toolset.objects")

    @model_validator(mode="after")
    def validate_visibility(self) -> "ToolsetConfig":
        if self.show is not None and self.hide is not None:
            raise ValueError("Toolset accepts show or hide, not both.")
        return self


class TasksetConfig(Config):
    supports_config_ref: ClassVar[bool] = True

    # Singleton fields describe one logical value owned by the taskset.
    source: TaskSource | None = None
    eval_source: TaskSource | None = None
    taskset_id: str | None = None
    system_prompt: PromptInput | None = None
    user: Handler | str | ConfigMap | None = None
    bindings: Bindings = Field(default_factory=dict)
    objects: Objects = Field(default_factory=dict)

    # Collection fields are merged/extended from code and config.
    toolsets: ToolsetSpecs | None = Field(default_factory=list)
    stops: list[CallableConfigEntry] = Field(default_factory=list)
    setups: list[CallableConfigEntry] = Field(default_factory=list)
    updates: list[CallableConfigEntry] = Field(default_factory=list)
    metrics: list[CallableConfigEntry] = Field(default_factory=list)
    rewards: list[CallableConfigEntry] = Field(default_factory=list)
    advantages: list[CallableConfigEntry] = Field(default_factory=list)
    cleanups: list[CallableConfigEntry] = Field(default_factory=list)
    scoring: dict[str, ConfigData] = Field(default_factory=dict)

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> Bindings:
        return normalize_binding_map(value, "taskset.bindings")

    @field_validator("objects", mode="before")
    @classmethod
    def validate_objects(cls, value: object) -> Objects:
        return normalize_object_map(value, "taskset.objects")


class HarnessConfig(Config):
    supports_config_ref: ClassVar[bool] = True

    # Singleton fields describe one logical value owned by the harness.
    program: Handler | str | ConfigMap | None = None
    system_prompt: PromptInput | None = None
    system_prompt_merge: str = "reject"
    sandbox: SandboxConfig | None = None
    client: ModelClient | ConfigMap | str | None = None
    model: str | None = None
    sampling_args: ConfigData = Field(default_factory=dict)
    keep_trajectory_step: Handler | str | None = None
    user: Handler | str | ConfigMap | None = None
    bindings: Bindings = Field(default_factory=dict)

    # Collection fields are merged/extended from code and config.
    toolsets: ToolsetSpecs | None = Field(default_factory=list)
    stops: list[CallableConfigEntry] = Field(default_factory=list)
    setups: list[CallableConfigEntry] = Field(default_factory=list)
    updates: list[CallableConfigEntry] = Field(default_factory=list)
    metrics: list[CallableConfigEntry] = Field(default_factory=list)
    rewards: list[CallableConfigEntry] = Field(default_factory=list)
    advantages: list[CallableConfigEntry] = Field(default_factory=list)
    cleanups: list[CallableConfigEntry] = Field(default_factory=list)
    scoring: dict[str, ConfigData] = Field(default_factory=dict)
    max_turns: int = 10

    @field_validator("program", mode="before")
    @classmethod
    def validate_program(cls, value: object) -> object:
        if value is None or callable(value) or isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            return ProgramConfig.from_config(
                string_mapping(cast(ConfigInputMap, value))
            ).model_dump(exclude_none=True, exclude_defaults=True)
        raise TypeError("program must be a callable, import ref, or mapping.")

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> Bindings:
        return normalize_binding_map(value, "harness.bindings", allow_objects=False)


class EnvConfig(Config):
    taskset: TasksetConfig = Field(default_factory=TasksetConfig)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)

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

    @classmethod
    def _merge_config_data(
        cls, config: ConfigSource | None, data: ConfigData
    ) -> ConfigData:
        data = dict(data)
        if config is None:
            return data
        base = config_data(config, cls)
        for section in ("taskset", "harness"):
            default_provided = section in data
            override_provided = section in base
            default = data.get(section)
            override = base.pop(section, None)
            if default_provided:
                if override_provided:
                    if default is None or override is None:
                        data[section] = None
                    else:
                        data[section] = merge_child_config(default, override)
                continue
            if override_provided:
                data[section] = override
        base.update(data)
        return base


def sandbox_config_mapping(value: object | None) -> ConfigData | None:
    if value is None:
        return None
    if isinstance(value, SandboxConfig):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        mapping = cast(ConfigMap, value)
        prefer = mapping.get("prefer")
        if prefer is not None and prefer != "program":
            raise ValueError("sandbox.prefer must be 'program'.")
        return SandboxConfig.from_config(mapping).model_dump(exclude_none=True)
    raise TypeError("Sandbox config must be a mapping.")
