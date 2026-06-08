import importlib
import inspect
import logging
import sys
from collections.abc import Mapping
from types import ModuleType, UnionType
from typing import (
    Callable,
    TypeAlias,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel
from verifiers.envs.environment import Environment
from verifiers.utils.config_utils import MissingKeyError
from verifiers.v1.env import Env, EnvConfig
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.types import ConfigData, ConfigValue
from verifiers.v1.utils.config_utils import coerce_config, explicit_config_data

EnvConfigLoadData: TypeAlias = dict[str, ConfigValue | BaseModel]
EnvConfigChildInput: TypeAlias = ConfigData | EnvConfigLoadData
EnvConfigInput: TypeAlias = EnvConfig | ConfigData


def load_environment(env_id: str, **env_args) -> Environment:
    logger = logging.getLogger("verifiers.utils.env_utils")
    logger.info(f"Loading environment: {env_id}")

    module_name = env_module_name(env_id)
    try:
        module = import_env_module(env_id)

        env_load_func: Callable[..., Environment] | None = getattr(
            module, "load_environment", None
        )
        if env_load_func is None:
            env_instance = load_environment_from_components(module, env_args)
            env_instance.env_id = env_instance.env_id or env_id
            env_instance.env_args = env_instance.env_args or env_args
            logger.info(f"Successfully loaded environment '{env_id}'")
            return env_instance

        sig = inspect.signature(env_load_func)
        defaults_info = []
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                if isinstance(param.default, (dict, list)):
                    defaults_info.append(f"{param_name}={param.default}")
                elif isinstance(param.default, str):
                    defaults_info.append(f"{param_name}='{param.default}'")
                else:
                    defaults_info.append(f"{param_name}={param.default}")
            else:
                defaults_info.append(f"{param_name}=<required>")

        if defaults_info:
            logger.debug(f"Environment defaults: {', '.join(defaults_info)}")

        if env_args:
            provided_params = set(env_args.keys())
        else:
            provided_params = set()

        all_params = set(sig.parameters.keys())
        default_params = all_params - provided_params

        if provided_params:
            provided_values = []
            for param_name in provided_params:
                provided_values.append(f"{param_name}={env_args[param_name]}")
            logger.info(f"Using provided args: {', '.join(provided_values)}")

        if default_params:
            default_values = []
            for param_name in default_params:
                param = sig.parameters[param_name]
                if param.default != inspect.Parameter.empty:
                    if isinstance(param.default, str):
                        default_values.append(f"{param_name}='{param.default}'")
                    else:
                        default_values.append(f"{param_name}={param.default}")
            if default_values:
                logger.info(f"Using default args: {', '.join(default_values)}")

        call_env_args = prepare_typed_env_config(module, env_load_func, sig, env_args)
        env_instance: Environment = env_load_func(**call_env_args)
        env_instance.env_id = env_instance.env_id or env_id
        env_instance.env_args = env_instance.env_args or env_args

        logger.info(f"Successfully loaded environment '{env_id}'")

        return env_instance

    except ImportError as e:
        logger.error(
            f"Failed to import environment module {module_name} for env_id {env_id}: {str(e)}"
        )
        raise ValueError(
            f"Could not import '{env_id}' environment. Ensure the package for the '{env_id}' environment is installed."
        ) from e
    except MissingKeyError:
        raise
    except Exception as e:
        logger.error(
            f"Failed to load environment {env_id} with args {env_args}: {str(e)}"
        )
        raise RuntimeError(f"Failed to load environment '{env_id}': {str(e)}") from e


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
    config: TasksetConfig | ConfigData | None = None,
) -> Taskset:
    module = caller_module() if env_id is None else import_env_module(env_id)
    return load_taskset_from_module(module, config=config)


def load_harness(
    env_id: str | None = None,
    *,
    config: HarnessConfig | ConfigData | None = None,
) -> Harness:
    module = caller_module() if env_id is None else import_env_module(env_id)
    return load_harness_from_module(module, config=config)


def load_taskset_from_module(
    module: ModuleType,
    *,
    config: TasksetConfig | ConfigData | None = None,
) -> Taskset:
    factory = getattr(module, "load_taskset", None)
    if factory is None:
        taskset_id = child_loader_id(config, "taskset_id")
        if taskset_id is not None and env_module_name(taskset_id) != module.__name__:
            return load_taskset(taskset_id, config=config)
        raise AttributeError(
            f"Module '{module.__name__}' does not expose load_taskset, and "
            "config.taskset_id is not set to a taskset loader package."
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
    config: HarnessConfig | ConfigData | None = None,
) -> Harness:
    factory = getattr(module, "load_harness", None)
    if factory is None:
        harness_id = child_loader_id(config, "harness_id")
        if harness_id is not None:
            if env_module_name(harness_id) == module.__name__:
                raise AttributeError(
                    f"Module '{module.__name__}' does not expose load_harness."
                )
            return load_harness(harness_id, config=config)
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


def prepare_typed_env_config(
    module: ModuleType,
    env_load_func: Callable[..., Environment],
    sig: inspect.Signature,
    env_args: dict,
) -> dict:
    config_type = env_config_annotation(env_load_func, sig)
    if config_type is None:
        return env_args

    config = env_args.get("config", {})
    if config is None:
        raise TypeError("load_environment config must be a concrete EnvConfig object.")

    call_env_args = dict(env_args)
    call_env_args["config"] = load_env_config(module, config_type, config)
    return call_env_args


def load_environment_from_components(
    module: ModuleType,
    env_args: dict,
) -> Env:
    extra_args = set(env_args) - {"config"}
    if extra_args:
        raise TypeError(
            "Default Taskset/Harness environment loading only accepts config; "
            f"got {sorted(extra_args)}."
        )
    config = load_env_config(module, EnvConfig, env_args.get("config", {}))
    return Env(
        taskset=load_taskset_from_module(module, config=config.taskset),
        harness=load_harness_from_module(module, config=config.harness),
    )


def env_config_annotation(
    env_load_func: Callable[..., Environment],
    sig: inspect.Signature,
) -> type[EnvConfig] | None:
    if "config" not in sig.parameters:
        return None
    try:
        annotation = get_type_hints(env_load_func).get(
            "config", sig.parameters["config"].annotation
        )
    except Exception:
        annotation = sig.parameters["config"].annotation
    return env_config_type(annotation)


def env_config_type(annotation: object) -> type[EnvConfig] | None:
    if annotation is inspect.Parameter.empty:
        return None
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            annotation = args[0]
    if isinstance(annotation, type) and issubclass(annotation, EnvConfig):
        return annotation
    return None


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
    defaults: EnvConfig | None = None
    for field_name, child_type in resolved_child_types.items():
        if field_name not in data:
            defaults = config_type() if defaults is None else defaults
            child = getattr(defaults, field_name)
            data[field_name] = child if isinstance(child, child_type) else child_type()
            continue
        child = data[field_name]
        if isinstance(child, child_type):
            continue
        if child is None:
            raise TypeError(f"config.{field_name} cannot be None.")
        if not isinstance(child, BaseModel | dict):
            raise TypeError(f"config.{field_name} must be a mapping or config object.")
        data[field_name] = child_type.model_validate(explicit_config_data(child))
    config = config_type.model_validate(data)
    for field_name, child_type in resolved_child_types.items():
        child = getattr(config, field_name)
        if not isinstance(child, child_type):
            raise TypeError(
                f"config.{field_name} must be {child_type.__name__}; "
                f"got {type(child).__name__}."
            )
    return config


def env_config_child_types(
    module: ModuleType,
    config_type: type[EnvConfig],
    value: EnvConfigChildInput | None = None,
) -> dict[str, type[BaseModel]]:
    child_types: dict[str, type[BaseModel]] = {}
    for field_name, id_field, factory_name, base_type in (
        ("taskset", "taskset_id", "load_taskset", TasksetConfig),
        ("harness", "harness_id", "load_harness", HarnessConfig),
    ):
        field_type = config_type_from_annotation(
            config_type.model_fields[field_name].annotation,
            base_type,
            f"{config_type.__name__}.{field_name}",
        )
        factory_type = factory_config_type(module, factory_name, base_type)
        child_config = value.get(field_name) if value is not None else None
        if (
            factory_type is None
            and field_type is base_type
            and child_config_requires_loader_type(child_config, base_type)
        ):
            loader_id = child_loader_id(child_config, id_field)
            if loader_id is not None and env_module_name(loader_id) != module.__name__:
                factory_type = factory_config_type(
                    import_env_module(loader_id), factory_name, base_type
                )
        if factory_type is not None:
            if not issubclass(factory_type, field_type):
                raise TypeError(
                    f"{module.__name__}.{factory_name} config type "
                    f"{factory_type.__name__} does not match "
                    f"{config_type.__name__}.{field_name}: {field_type.__name__}."
                )
            child_types[field_name] = factory_type
        else:
            child_types[field_name] = field_type
    return child_types


def child_config_requires_loader_type(
    config: object,
    base_type: type[BaseModel],
) -> bool:
    if not isinstance(config, Mapping):
        return False
    base_fields = set(base_type.model_fields) | {"id"}
    return bool(set(config) - base_fields)


def child_loader_id(config: object, id_field: str) -> str | None:
    if isinstance(config, BaseModel):
        value = config.__dict__.get(id_field)
    elif isinstance(config, Mapping):
        config_data = dict(config)
        value = config_data.get(id_field) or config_data.get("id")
    else:
        return None
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"config.{id_field} must be a non-empty string.")
    return value


def factory_config_type(
    module: ModuleType,
    factory_name: str,
    base_type: type[BaseModel],
) -> type[BaseModel] | None:
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
