from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Mapping
from types import ModuleType
from typing import (
    Callable,
    TYPE_CHECKING,
    TypeAlias,
)

from pydantic import BaseModel
from verifiers.envs.environment import Environment
from verifiers.utils.config_utils import MissingKeyError

if TYPE_CHECKING:
    from verifiers.v1.env import Env, EnvConfig
    from verifiers.v1.harness import Harness, HarnessConfig
    from verifiers.v1.taskset import Taskset, TasksetConfig

ConfigMapping: TypeAlias = Mapping[str, object]
EnvConfigLoadData: TypeAlias = dict[str, object]
EnvConfigChildInput: TypeAlias = ConfigMapping | EnvConfigLoadData
EnvConfigInput: TypeAlias = BaseModel | ConfigMapping


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


def load_taskset(
    env_id: str | None = None,
    *,
    config: TasksetConfig | ConfigMapping | None = None,
) -> Taskset:
    from verifiers.v1.loaders import load_taskset as load_v1_taskset

    return load_v1_taskset(env_id, config=config)


def load_harness(
    env_id: str | None = None,
    *,
    config: HarnessConfig | ConfigMapping | None = None,
) -> Harness:
    from verifiers.v1.loaders import load_harness as load_v1_harness

    return load_v1_harness(env_id, config=config)


def load_taskset_from_module(
    module: ModuleType,
    *,
    config: TasksetConfig | ConfigMapping | None = None,
) -> Taskset:
    from verifiers.v1.loaders import load_taskset_from_module as load_v1_taskset

    return load_v1_taskset(module, config=config)


def load_harness_from_module(
    module: ModuleType,
    *,
    config: HarnessConfig | ConfigMapping | None = None,
) -> Harness:
    from verifiers.v1.loaders import load_harness_from_module as load_v1_harness

    return load_v1_harness(module, config=config)


def prepare_typed_env_config(
    module: ModuleType,
    env_load_func: Callable[..., Environment],
    sig: inspect.Signature,
    env_args: dict,
) -> dict:
    from verifiers.v1.loaders import prepare_typed_env_config as prepare_v1_config

    return prepare_v1_config(module, env_load_func, sig, env_args)


def load_environment_from_components(
    module: ModuleType,
    env_args: dict,
) -> Env:
    from verifiers.v1.loaders import (
        load_environment_from_components as load_v1_components,
    )

    return load_v1_components(module, env_args)


def env_config_annotation(
    env_load_func: Callable[..., Environment],
    sig: inspect.Signature,
) -> type[EnvConfig] | None:
    from verifiers.v1.loaders import env_config_annotation as v1_config_annotation

    return v1_config_annotation(env_load_func, sig)


def env_config_type(annotation: object) -> type[EnvConfig] | None:
    from verifiers.v1.loaders import env_config_type as v1_config_type

    return v1_config_type(annotation)


def load_env_config(
    module: ModuleType,
    config_type: type[EnvConfig],
    value: EnvConfigInput,
    *,
    child_types: Mapping[str, type[BaseModel]] | None = None,
) -> EnvConfig:
    from verifiers.v1.loaders import load_env_config as load_v1_env_config

    return load_v1_env_config(
        module,
        config_type,
        value,
        child_types=child_types,
    )


def env_config_child_types(
    module: ModuleType,
    config_type: type[EnvConfig],
    value: EnvConfigChildInput | None = None,
) -> dict[str, type[BaseModel]]:
    from verifiers.v1.loaders import env_config_child_types as v1_child_types

    return v1_child_types(module, config_type, value)


def child_config_requires_loader_type(
    config: object,
    base_type: type[BaseModel],
) -> bool:
    from verifiers.v1.loaders import (
        child_config_requires_loader_type as v1_requires_loader_type,
    )

    return v1_requires_loader_type(config, base_type)


def child_loader_id(config: object) -> str | None:
    from verifiers.v1.loaders import child_loader_id as v1_child_loader_id

    return v1_child_loader_id(config)


def factory_config_type(
    module: ModuleType,
    factory_name: str,
    base_type: type[BaseModel],
) -> type[BaseModel] | None:
    from verifiers.v1.loaders import factory_config_type as v1_factory_config_type

    return v1_factory_config_type(module, factory_name, base_type)


def config_type_from_annotation(
    annotation: object,
    base_type: type[BaseModel],
    context: str,
) -> type[BaseModel]:
    from verifiers.v1.loaders import config_type_from_annotation as v1_config_type

    return v1_config_type(annotation, base_type, context)
