from __future__ import annotations

import functools
import importlib
import inspect
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticUndefined
from typing_extensions import Self

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def __init__(self, config: object | None = None, /, **data: object):
        super().__init__(**type(self)._merge_config_data(config, data))

    @classmethod
    def from_config(cls, config: object | None = None, /, **data: object) -> Self:
        return cls(**cls._merge_config_data(config, data))

    @classmethod
    def _merge_config_data(
        cls, config: object | None, data: dict[str, object]
    ) -> dict[str, object]:
        data = omit_none(data)
        if config is not None:
            base = config_data(config, cls)
            base.update(data)
            data = base
        return data

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


def config_data(value: object, target: type[Config] | None = None) -> dict[str, object]:
    if value is None:
        data: dict[str, object] = {}
    elif isinstance(value, Config):
        data = value.model_dump(exclude_none=True, exclude_unset=True)
        if target is not None:
            data = {
                key: item for key, item in data.items() if key in target.model_fields
            }
    elif isinstance(value, Mapping):
        data = string_mapping(cast(Mapping[object, object], value))
    else:
        raise TypeError("Config must be a mapping or Config object.")
    return data


def omit_none(data: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in data.items() if value is not None}


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


class UserConfig(Config):
    fn: object
    scope: Literal["rollout", "group", "global"] = "rollout"
    bindings: dict[str, object] = Field(default_factory=dict)
    objects: dict[str, object] = Field(default_factory=dict)
    sandbox: SandboxConfig | None = None


class ToolsetConfig(Config):
    tools: object = Field(default_factory=list)
    show: list[str] | None = None
    hide: list[str] | None = None
    bindings: dict[str, object] = Field(default_factory=dict)
    objects: dict[str, object] = Field(default_factory=dict)
    write: bool = False
    scope: Literal["rollout", "group", "global"] | None = None
    sandbox: SandboxConfig | Literal["program"] | None = None
    stops: list[object] = Field(default_factory=list)
    setups: list[object] = Field(default_factory=list)
    updates: list[object] = Field(default_factory=list)
    cleanups: list[object] = Field(default_factory=list)
    teardowns: list[object] = Field(default_factory=list)

    @field_validator("show", "hide", mode="before")
    @classmethod
    def validate_visibility_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def validate_visibility(self) -> ToolsetConfig:
        if self.show is not None and self.hide is not None:
            raise ValueError("Toolset accepts show or hide, not both.")
        return self


def validate_import_ref(import_ref: str) -> str:
    module_name, separator, attr_path = import_ref.partition(":")
    if not separator or not module_name or not attr_path:
        raise ValueError(
            f"Import ref {import_ref!r} must use 'module:object' (e.g. 'pkg.mod:fn')."
        )
    return import_ref


ImportRef = Annotated[str, AfterValidator(validate_import_ref)]
"""A ``module.submodule:object`` import ref. Validated at config-construction
time; only imported when the config is materialized into a callable."""


class CallableConfig(Config):
    """Typed config spec for a callable contributed via TOML or dict.

    ``fn`` is an import ref (``"pkg.mod:obj"``). Raw callables are not
    valid here — pass them as constructor args or define them as
    decorated class methods on a Taskset/Harness subclass. Bare-string
    list entries are auto-promoted to ``{"fn": "..."}``.
    """

    fn: ImportRef
    priority: int = 0

    @model_validator(mode="before")
    @classmethod
    def _coerce_bare(cls, data: object) -> object:
        if isinstance(data, cls):
            return data
        if isinstance(data, str):
            return {"fn": data}
        return data

    def import_fn(self) -> Callable[..., object]:
        """Import ``self.fn`` and return the underlying callable.

        Raises ``TypeError`` if the imported object is not callable, and
        ``ModuleNotFoundError`` / ``AttributeError`` if the ref does not
        resolve (the validator only checks the ref's *shape*, not whether
        the target module/attribute exists).
        """
        obj = import_config_ref(self.fn)
        if not callable(obj):
            raise TypeError(
                f"Import ref {self.fn!r} resolved to {type(obj).__name__}, "
                f"expected a callable."
            )
        return cast(Callable[..., object], obj)


class SignalConfig(CallableConfig):
    stage: Literal["rollout", "group"] = "rollout"


class MetricConfig(SignalConfig):
    weight: float = 1.0


class RewardConfig(SignalConfig):
    weight: float = 1.0


class TasksetConfig(Config):
    # Singleton fields describe one logical value owned by the taskset.
    source: object | None = None
    eval_source: object | None = None
    taskset_id: str | None = None
    system_prompt: object | None = None
    user: object | None = None

    # Collection fields are merged/extended from code and config.
    toolsets: object = Field(default_factory=list)
    stops: list[CallableConfig] = Field(default_factory=list)
    setups: list[CallableConfig] = Field(default_factory=list)
    updates: list[SignalConfig] = Field(default_factory=list)
    metrics: list[MetricConfig] = Field(default_factory=list)
    rewards: list[RewardConfig] = Field(default_factory=list)
    advantages: list[CallableConfig] = Field(default_factory=list)
    cleanups: list[SignalConfig] = Field(default_factory=list)
    scoring: dict[str, dict[str, object]] = Field(default_factory=dict)


class HarnessConfig(Config):
    """Configures a harness."""

    # Singleton fields describe one logical value owned by the harness.
    program: object | None = None
    system_prompt: object | None = None
    system_prompt_merge: str = "reject"
    sandbox: SandboxConfig | None = None
    client: object | None = None
    model: str | None = None
    sampling_args: dict[str, object] = Field(default_factory=dict)
    keep_trajectory_step: object | None = None
    user: object | None = None

    # Collection fields are merged/extended from code and config.
    toolsets: object = Field(default_factory=list)
    stops: list[CallableConfig] = Field(default_factory=list)
    setups: list[CallableConfig] = Field(default_factory=list)
    updates: list[SignalConfig] = Field(default_factory=list)
    metrics: list[SignalConfig] = Field(default_factory=list)
    rewards: list[RewardConfig] = Field(default_factory=list)
    advantages: list[CallableConfig] = Field(default_factory=list)
    cleanups: list[SignalConfig] = Field(default_factory=list)
    scoring: dict[str, dict[str, object]] = Field(default_factory=dict)
    max_turns: int = 10


class EnvConfig(Config):
    taskset: object | None = None
    harness: object | None = None

    @field_validator("taskset", "harness")
    @classmethod
    def validate_child_config(cls, value: object) -> object:
        if value is not None:
            try:
                config_data(value)
            except TypeError as exc:
                raise ValueError(str(exc)) from exc
        return value

    @classmethod
    def _merge_config_data(
        cls, config: object | None, data: dict[str, object]
    ) -> dict[str, object]:
        data = omit_none(data)
        if config is None:
            return data
        base = config_data(config, cls)
        for section in ("taskset", "harness"):
            default = data.get(section)
            override = base.pop(section, None)
            if default is None:
                if override is not None:
                    data[section] = override
                continue
            if override is not None:
                data[section] = merge_child_config(default, override)
        base.update(data)
        return base


def merge_child_config(default: object, override: object) -> object:
    merged = deep_merge(config_data(default), config_data(override))
    if isinstance(default, Config):
        return type(default)(merged)
    return merged


def merge_config_value(value: object, config: object) -> object:
    if config is None:
        return value
    if value is None:
        return config
    value_mapping = config_mapping(value)
    config_mapping_value = config_mapping(config)
    if value_mapping is not None and config_mapping_value is not None:
        return deep_merge(
            config_mapping_value,
            value_mapping,
        )
    return value


def config_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, Config):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return string_mapping(cast(Mapping[object, object], value))
    return None


def sandbox_config_mapping(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, SandboxConfig):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        prefer = mapping.get("prefer")
        if prefer is not None and prefer != "program":
            raise ValueError("sandbox.prefer must be 'program'.")
        return SandboxConfig.from_config(mapping).model_dump(exclude_none=True)
    raise TypeError("Sandbox config must be a mapping.")


def merge_config_items(values: Iterable[object], config: object) -> list[object]:
    return [*values, *config_items(config)]


CallableKind = Literal[
    "stop", "setup", "update", "metric", "reward", "advantage", "cleanup", "teardown"
]


def merge_config_callables(
    values: Iterable[Callable[..., object]],
    config: object,
    kind: CallableKind,
) -> list[Callable[..., object]]:
    return [*config_callables(values, kind), *config_callables(config, kind)]


def config_callables(value: object, kind: CallableKind) -> list[Callable[..., object]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [callable_config_item(value, kind)]
    if isinstance(value, Mapping):
        return [callable_config_item(value, kind)]
    if isinstance(value, Iterable):
        return [callable_config_item(item, kind) for item in value]
    return [callable_config_item(value, kind)]


def callable_config_item(value: object, kind: CallableKind) -> Callable[..., object]:
    if isinstance(value, CallableConfig):
        return _resolve_callable_config(value, kind)
    value = resolve_config_object(value)
    if isinstance(value, Mapping):
        return callable_from_mapping(cast(Mapping[str, object], value), kind)
    if not callable(value):
        raise TypeError(f"{kind} config entries must resolve to callables.")
    return cast(Callable[..., object], value)


def _resolve_callable_config(
    spec: CallableConfig, kind: CallableKind
) -> Callable[..., object]:
    fn = spec.import_fn()
    explicit = spec.model_fields_set
    metadata: dict[str, object] = {}
    if "priority" in explicit:
        metadata["priority"] = spec.priority
    if isinstance(spec, SignalConfig) and "stage" in explicit:
        metadata["stage"] = spec.stage
    if isinstance(spec, RewardConfig) and "weight" in explicit:
        metadata["weight"] = spec.weight
    return configured_callable(cast(Callable[..., object], fn), kind, metadata)


def callable_from_mapping(
    spec: Mapping[str, object], kind: CallableKind
) -> Callable[..., object]:
    allowed = callable_config_keys(kind)
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"{kind} callable config has unknown keys: {sorted(unknown)}.")
    if bool(spec.get("skip", False)):
        raise ValueError(
            f"{kind} callable config should be removed instead of skipped."
        )
    fn = resolve_config_object(spec.get("fn"))
    if not callable(fn):
        raise TypeError(f"{kind} callable config requires callable fn.")
    metadata = {key: spec[key] for key in spec if key not in {"fn", "skip"}}
    return configured_callable(cast(Callable[..., object], fn), kind, metadata)


def callable_config_keys(kind: CallableKind) -> set[str]:
    keys = {"fn", "priority", "skip"}
    if kind in {"update", "metric", "reward", "cleanup"}:
        keys.add("stage")
    if kind == "reward":
        keys.add("weight")
    return keys


def configured_callable(
    fn: Callable[..., object],
    kind: CallableKind,
    metadata: Mapping[str, object],
) -> Callable[..., object]:
    if not metadata:
        return fn

    @functools.wraps(fn)
    async def wrapper(**kwargs: object) -> object:
        result = fn(**kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    setattr(wrapper, "__signature__", inspect.signature(fn))
    setattr(wrapper, kind, True)
    if "priority" in metadata:
        priority = metadata["priority"]
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError(f"{kind} priority must be an integer.")
        setattr(wrapper, f"{kind}_priority", priority)
    if "stage" in metadata:
        stage = metadata["stage"]
        if stage not in {"rollout", "group"}:
            raise ValueError(f"{kind} stage must be 'rollout' or 'group'.")
        setattr(wrapper, f"{kind}_stage", stage)
    if "weight" in metadata:
        weight = metadata["weight"]
        if not isinstance(weight, int | float) or isinstance(weight, bool):
            raise TypeError("reward weight must be numeric.")
        setattr(wrapper, "reward_weight", float(weight))
    return cast(Callable[..., object], wrapper)


def config_items(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [import_config_ref(value)]
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Iterable):
        return [resolve_config_object(item) for item in value]
    return [value]


def resolve_config_object(value: object) -> object:
    if isinstance(value, str):
        return import_config_ref(value)
    return value


def import_config_ref(ref: str) -> object:
    module_name, separator, attr_path = ref.partition(":")
    if not separator or not module_name or not attr_path:
        raise ValueError(f"Config ref {ref!r} must use 'module:object'.")
    obj: object = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def deep_merge(
    base: dict[str, object], overlay: Mapping[str, object]
) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(
                string_mapping(cast(Mapping[object, object], existing)),
                string_mapping(cast(Mapping[object, object], value)),
            )
        else:
            merged[key] = value
    return merged


def string_mapping(value: Mapping[object, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Config mappings require string keys.")
        result[key] = item
    return result


def annotation_text(annotation: Any) -> str:
    if getattr(annotation, "__args__", None):
        return str(annotation).replace("typing.", "")
    name = getattr(annotation, "__name__", None)
    if isinstance(name, str):
        return name
    return str(annotation).replace("typing.", "")


def default_text(field: object) -> str:
    default_factory = getattr(field, "default_factory", None)
    if default_factory is not None:
        return "<factory>"
    default = getattr(field, "default", PydanticUndefined)
    if default is PydanticUndefined:
        return "required"
    return repr(default)
