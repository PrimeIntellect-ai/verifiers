from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from collections.abc import Mapping
from types import ModuleType, UnionType
from typing import TypeAlias, Union, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from .env import Env, EnvConfig
from .harness import Harness, HarnessConfig
from .taskset import Taskset, TasksetConfig
from .utils.config_utils import coerce_config, explicit_config_data

ConfigMapping: TypeAlias = Mapping[str, object]
EnvConfigLoadData: TypeAlias = dict[str, object]
EnvConfigChildInput: TypeAlias = ConfigMapping | EnvConfigLoadData
EnvConfigInput: TypeAlias = BaseModel | ConfigMapping

FACTORY_MODULES = {
    "load_taskset": "taskset",
    "load_harness": "harness",
}


def env_module_name(env_id: str) -> str:
    return env_id.replace("-", "_").split("/")[-1]


def import_env_module(env_id: str) -> ModuleType:
    return importlib.import_module(env_module_name(env_id))


def caller_module() -> ModuleType:
    frame = inspect.currentframe()
    try:
        if frame is None or frame.f_back is None or frame.f_back.f_back is None:
            raise RuntimeError("Could not resolve caller module.")
        module_name = frame.f_back.f_back.f_globals.get("__name__")
        if not isinstance(module_name, str):
            raise RuntimeError("Caller module has no __name__.")
        module = sys.modules.get(module_name)
        if not isinstance(module, ModuleType):
            raise RuntimeError(f"Caller module {module_name!r} is not loaded.")
        return module
    finally:
        del frame


def load_taskset(
    env_id: str | None = None,
    *,
    config: TasksetConfig | ConfigMapping | None = None,
) -> Taskset:
    module = caller_module() if env_id is None else import_env_module(env_id)
    return load_taskset_from_module(module, config=config)


def load_harness(
    env_id: str | None = None,
    *,
    config: HarnessConfig | ConfigMapping | None = None,
) -> Harness:
    module = caller_module() if env_id is None else import_env_module(env_id)
    return load_harness_from_module(module, config=config)


def load_environment(env_id: str, **env_args: object) -> Env:
    return load_environment_from_components(import_env_module(env_id), env_args)


def load_taskset_from_module(
    module: ModuleType,
    *,
    config: TasksetConfig | ConfigMapping | None = None,
) -> Taskset:
    source_module_name = module.__name__
    module = factory_module(module, "load_taskset")
    factory = getattr(module, "load_taskset", None)
    if factory is None:
        loader_id = child_loader_id(config)
        if loader_id is not None and not matches_loader(source_module_name, loader_id):
            return load_taskset(loader_id, config=config)
        raise AttributeError(
            f"Module '{module.__name__}' does not expose load_taskset, and "
            "config.id is not set to a taskset loader package."
        )
    config_type = factory_config_type(module, "load_taskset", TasksetConfig)
    if config_type is None:
        raise TypeError(f"{module.__name__}.load_taskset must accept config.")
    taskset = factory(
        config=coerce_config(cast(type[TasksetConfig], config_type), config)
    )
    if not isinstance(taskset, Taskset):
        raise TypeError(f"{module.__name__}.load_taskset must return a Taskset.")
    return taskset


def load_harness_from_module(
    module: ModuleType,
    *,
    config: HarnessConfig | ConfigMapping | None = None,
) -> Harness:
    source_module_name = module.__name__
    module = factory_module(module, "load_harness")
    factory = getattr(module, "load_harness", None)
    if factory is None:
        loader_id = child_loader_id(config)
        if loader_id is not None:
            if matches_loader(source_module_name, loader_id):
                raise AttributeError(
                    f"Module '{module.__name__}' does not expose load_harness."
                )
            return load_harness(loader_id, config=config)
        return Harness(config=coerce_config(HarnessConfig, config))
    config_type = factory_config_type(module, "load_harness", HarnessConfig)
    if config_type is None:
        raise TypeError(f"{module.__name__}.load_harness must accept config.")
    harness = factory(
        config=coerce_config(cast(type[HarnessConfig], config_type), config)
    )
    if not isinstance(harness, Harness):
        raise TypeError(f"{module.__name__}.load_harness must return a Harness.")
    return harness


def load_environment_from_components(
    module: ModuleType,
    env_args: dict[str, object],
) -> Env:
    extra_args = set(env_args) - {"config"}
    if extra_args:
        raise TypeError(
            "Default Taskset/Harness environment loading only accepts config; "
            f"got {sorted(extra_args)}."
        )
    config_input = env_args.get("config", {})
    if not isinstance(config_input, BaseModel | Mapping):
        raise TypeError("config must be a mapping or EnvConfig.")
    config = load_env_config(module, EnvConfig, cast(EnvConfigInput, config_input))
    return Env(
        taskset=load_taskset_from_module(module, config=config.taskset),
        harness=load_harness_from_module(module, config=config.harness),
        runtime=config.runtime,
    )


def load_env_config(
    module: ModuleType,
    config_type: type[EnvConfig],
    value: EnvConfigInput,
    *,
    child_types: Mapping[str, type[BaseModel]] | None = None,
) -> EnvConfig:
    data: EnvConfigLoadData
    if isinstance(value, config_type):
        data = dict(explicit_config_data(value))
    elif isinstance(value, BaseModel):
        raise TypeError(
            f"load_environment config must be {config_type.__name__}; "
            f"got {type(value).__name__}."
        )
    elif not isinstance(value, Mapping):
        raise TypeError("load_environment config must be a mapping or EnvConfig.")
    else:
        data = dict(value)
    resolved_child_types = (
        env_config_child_types(module, config_type, data)
        if child_types is None
        else child_types
    )
    for field_name, child_type in resolved_child_types.items():
        if field_name not in data:
            data[field_name] = child_type()
            continue
        child = data[field_name]
        if isinstance(child, child_type):
            continue
        if child is None:
            raise TypeError(f"config.{field_name} cannot be None.")
        if not isinstance(child, BaseModel | Mapping):
            raise TypeError(f"config.{field_name} must be a mapping or config object.")
        data[field_name] = child_type.model_validate(
            explicit_config_data(cast(EnvConfigInput, child))
        )
    return config_type.model_validate(data)


def env_config_child_types(
    module: ModuleType,
    config_type: type[EnvConfig],
    value: EnvConfigChildInput | None = None,
) -> dict[str, type[BaseModel]]:
    child_types: dict[str, type[BaseModel]] = {}
    for field_name, factory_name, base_type in (
        ("taskset", "load_taskset", TasksetConfig),
        ("harness", "load_harness", HarnessConfig),
    ):
        factory_type = factory_config_type(module, factory_name, base_type)
        child_config = value.get(field_name) if value is not None else None
        if factory_type is None and child_config_requires_loader_type(
            child_config, base_type
        ):
            loader_id = child_loader_id(child_config)
            if loader_id is not None and not matches_loader(module.__name__, loader_id):
                factory_type = factory_config_type(
                    import_env_module(loader_id), factory_name, base_type
                )
        if factory_type is not None:
            child_types[field_name] = factory_type
        else:
            child_types[field_name] = base_type
    return child_types


def child_config_requires_loader_type(
    config: object,
    base_type: type[BaseModel],
) -> bool:
    if not isinstance(config, Mapping):
        return False
    base_fields = set(base_type.model_fields)
    return bool(set(config) - base_fields)


def child_loader_id(config: object) -> str | None:
    if isinstance(config, BaseModel):
        value = config.__dict__.get("id")
    elif isinstance(config, Mapping):
        value = dict(config).get("id")
    else:
        return None
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError("config.id must be a non-empty string.")
    return value


def matches_loader(module_name: str, loader_id: str) -> bool:
    loader_module_name = env_module_name(loader_id)
    return module_name == loader_module_name or module_name.startswith(
        f"{loader_module_name}."
    )


def factory_config_type(
    module: ModuleType,
    factory_name: str,
    base_type: type[BaseModel],
) -> type[BaseModel] | None:
    module = factory_module(module, factory_name)
    factory = getattr(module, factory_name, None)
    if factory is None:
        return None
    signature = inspect.signature(factory)
    if "config" not in signature.parameters:
        raise TypeError(f"{module.__name__}.{factory_name} must accept config.")
    try:
        annotation = get_type_hints(factory).get(
            "config", signature.parameters["config"].annotation
        )
    except Exception:
        annotation = signature.parameters["config"].annotation
    return config_type_from_annotation(
        annotation,
        base_type,
        f"{module.__name__}.{factory_name}.config",
    )


def factory_module(module: ModuleType, factory_name: str) -> ModuleType:
    if getattr(module, factory_name, None) is not None:
        return module
    child_name = FACTORY_MODULES.get(factory_name)
    if child_name is None or not hasattr(module, "__path__"):
        return module
    module_name = f"{module.__name__}.{child_name}"
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return module
    return importlib.import_module(module_name)


def config_type_from_annotation(
    annotation: object,
    base_type: type[BaseModel],
    context: str,
) -> type[BaseModel]:
    if annotation is inspect.Parameter.empty:
        raise TypeError(f"{context} must be annotated.")
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            annotation = args[0]
    if isinstance(annotation, type) and issubclass(annotation, base_type):
        return annotation
    raise TypeError(f"{context} must be a {base_type.__name__} subclass.")
