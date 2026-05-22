from collections.abc import Mapping
from os import PathLike
from typing import Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)
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
    coerce_config,
    default_text,
    explicit_config_data,
    import_config_ref as import_config_ref,
    resolved_config_data,
    resolve_config_object as resolve_config_object,
    string_mapping,
)
from .utils.component_utils import (
    component_config_data,
    component_config_type,
    component_id_from_data,
    component_loader,
    import_component_module,
)
from .utils.mcp_proxy_utils import validate_program_channels
from verifiers.types import ClientConfig


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
        result[name] = coerce_config(SignalConfig, data).model_dump(
            exclude_none=True,
            exclude_unset=True,
        )
    return result


class LifecycleConfig(Config):
    # Collection fields are configured only here; runtime mutation APIs are separate.
    toolsets: ToolsetSpecs | None = []
    stops: list[CallableEntry] = []
    setups: list[CallableEntry] = []
    updates: list[CallableEntry] = []
    metrics: list[CallableEntry] = []
    rewards: list[CallableEntry] = []
    advantages: list[CallableEntry] = []
    cleanups: list[CallableEntry] = []
    teardowns: list[CallableEntry] = []
    scoring: dict[str, ConfigData] = {}

    @field_validator("scoring", mode="before")
    @classmethod
    def validate_scoring(cls, value: object) -> dict[str, ConfigData]:
        return validate_scoring_map(value, "scoring")


class TasksetConfig(LifecycleConfig):
    _vf_loader_id: str | None = PrivateAttr(default=None)

    # Singleton fields describe one logical value owned by the taskset.
    source: TaskSource | None = None
    eval_source: TaskSource | None = None
    taskset_id: str | None = None
    system_prompt: PromptInput | None = None
    user: UserConfig | str | None = None
    bindings: Bindings = {}
    objects: dict[str, str] = {}

    @field_validator("bindings", mode="before")
    @classmethod
    def validate_bindings(cls, value: object) -> Bindings:
        return normalize_binding_map(value, "taskset.bindings")


class HarnessConfig(LifecycleConfig):
    _vf_loader_id: str | None = PrivateAttr(default=None)

    # Singleton fields describe one logical value owned by the harness.
    harness_id: str | None = None
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
            if field_name not in cls.__dict__.get("__annotations__", {}):
                continue
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
            data = explicit_config_data(value)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        annotation = cls.model_fields[str(info.field_name)].annotation
        if (
            info.field_name == "taskset"
            and isinstance(annotation, type)
            and issubclass(annotation, TasksetConfig)
        ):
            if isinstance(value, annotation):
                return value
            component_id = component_id_from_data(
                data, alias_field="taskset_id", label="taskset"
            )
            validate_env_child_config(data, str(info.field_name))
            if component_id is None:
                return data
            if cls is not EnvConfig and "id" not in data:
                return data
            config_cls: type[BaseModel] = annotation
            module = import_component_module(component_id, "taskset")
            loader = component_loader(module, "load_taskset", component_id, "taskset")
            package_config_cls = component_config_type(
                loader=loader,
                loader_name="load_taskset",
                component_id=component_id,
                base_config_cls=TasksetConfig,
                label="taskset",
            )
            if annotation is not TasksetConfig and not issubclass(
                package_config_cls, annotation
            ):
                raise ValueError(
                    f"{cls.__name__}.taskset is typed as {annotation.__name__}, "
                    f"but taskset package {component_id!r} expects "
                    f"{package_config_cls.__name__}. Use base vf.EnvConfig for "
                    "package-selected tasksets."
                )
            config_cls = package_config_cls
            config = coerce_config(
                config_cls,
                component_config_data(
                    data=data,
                    component_id=component_id,
                    alias_field="taskset_id",
                    config_cls=config_cls,
                ),
            )
            cast(TasksetConfig, config)._vf_loader_id = component_id
            return config
        if (
            info.field_name == "harness"
            and isinstance(annotation, type)
            and issubclass(annotation, HarnessConfig)
        ):
            if isinstance(value, annotation):
                return value
            component_id = component_id_from_data(
                data, alias_field="harness_id", label="harness"
            )
            validate_env_child_config(data, str(info.field_name))
            if component_id is None:
                return data
            if cls is not EnvConfig and "id" not in data:
                return data
            config_cls = annotation
            module = import_component_module(component_id, "harness")
            loader = component_loader(module, "load_harness", component_id, "harness")
            package_config_cls = component_config_type(
                loader=loader,
                loader_name="load_harness",
                component_id=component_id,
                base_config_cls=HarnessConfig,
                label="harness",
            )
            if annotation is not HarnessConfig and not issubclass(
                package_config_cls, annotation
            ):
                raise ValueError(
                    f"{cls.__name__}.harness is typed as {annotation.__name__}, "
                    f"but harness package {component_id!r} expects "
                    f"{package_config_cls.__name__}. Use base vf.EnvConfig for "
                    "package-selected harnesses."
                )
            config_cls = package_config_cls
            config = coerce_config(
                config_cls,
                component_config_data(
                    data=data,
                    component_id=component_id,
                    alias_field="harness_id",
                    config_cls=config_cls,
                ),
            )
            cast(HarnessConfig, config)._vf_loader_id = component_id
            return config
        return data


def validate_env_child_config(data: ConfigData, field_name: str) -> None:
    if "config" in data:
        raise ValueError(
            f"EnvConfig.{field_name}.config is not supported. "
            "Put fields directly in the section."
        )
    for key, item in data.items():
        validate_serializable_value(item, f"EnvConfig.{field_name}.{key}")


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
