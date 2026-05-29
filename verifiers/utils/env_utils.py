import importlib
import inspect
import logging
import sys
from collections.abc import Mapping
from types import ModuleType, UnionType
from typing import Callable, Union, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel
from verifiers.envs.environment import Environment
from verifiers.utils.config_utils import MissingKeyError
from verifiers.v1.env import EnvConfig
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.utils.config_utils import coerce_config, explicit_config_data


def load_environment(env_id: str, **env_args) -> Environment:
    logger = logging.getLogger("verifiers.utils.env_utils")
    logger.info(f"Loading environment: {env_id}")

    module_name = env_module_name(env_id)
    try:
        module = import_env_module(env_id)

        if not hasattr(module, "load_environment"):
            raise AttributeError(
                f"Module '{module_name}' does not have a 'load_environment' function. "
                f"This usually means there's a package name collision. Please either:\n"
                f"1. Rename your environment (e.g. suffix with '-env')\n"
                f"2. Remove unneeded files with the same name\n"
                f"3. Check that you've installed the correct environment package"
            )

        env_load_func: Callable[..., Environment] = getattr(module, "load_environment")
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
    config: TasksetConfig | Mapping[str, object] | None = None,
) -> Taskset:
    module = caller_module() if env_id is None else import_env_module(env_id)
    return load_taskset_from_module(module, config=config)


def load_harness(
    env_id: str | None = None,
    *,
    config: HarnessConfig | Mapping[str, object] | None = None,
) -> Harness:
    module = caller_module() if env_id is None else import_env_module(env_id)
    return load_harness_from_module(module, config=config)


def load_taskset_from_module(
    module: ModuleType,
    *,
    config: TasksetConfig | Mapping[str, object] | None = None,
) -> Taskset:
    factory = getattr(module, "load_taskset", None)
    if factory is None:
        raise AttributeError(
            f"Module '{module.__name__}' does not expose load_taskset."
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
    config: HarnessConfig | Mapping[str, object] | None = None,
) -> Harness:
    factory = getattr(module, "load_harness", None)
    if factory is None:
        raise AttributeError(
            f"Module '{module.__name__}' does not expose load_harness."
        )
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
    value: object,
    *,
    child_types: Mapping[str, type[BaseModel]] | None = None,
) -> EnvConfig:
    resolved_child_types = (
        env_config_child_types(module, config_type)
        if child_types is None
        else child_types
    )

    data: dict[str, object]
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
        data = dict(explicit_config_data(value))
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
) -> dict[str, type[BaseModel]]:
    child_types: dict[str, type[BaseModel]] = {}
    for field_name, factory_name, base_type in (
        ("taskset", "load_taskset", TasksetConfig),
        ("harness", "load_harness", HarnessConfig),
    ):
        field_type = config_type_from_annotation(
            config_type.model_fields[field_name].annotation,
            base_type,
            f"{config_type.__name__}.{field_name}",
        )
        factory_type = factory_config_type(module, factory_name, base_type)
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
