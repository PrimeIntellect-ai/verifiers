import importlib
import inspect
import logging
from collections.abc import Mapping
from typing import Callable, get_type_hints

from pydantic import BaseModel
from verifiers.envs.environment import Environment
from verifiers.utils.config_utils import MissingKeyError
from verifiers.v1.config import EnvConfig
from verifiers.v1.utils.config_utils import coerce_config, explicit_config_data


def package_module_name(package_id: str) -> str:
    return package_id.replace("-", "_").split("/")[-1]


def load_environment(env_id: str, **env_args) -> Environment:
    logger = logging.getLogger("verifiers.utils.env_utils")
    logger.info(f"Loading environment: {env_id}")

    module_name = package_module_name(env_id)
    try:
        module = importlib.import_module(module_name)

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

        call_env_args = prepare_typed_env_config(
            env_load_func, sig, env_args, env_id=env_id
        )
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


def prepare_typed_env_config(
    env_load_func: Callable[..., Environment],
    sig: inspect.Signature,
    env_args: dict,
    *,
    env_id: str | None = None,
) -> dict:
    config_type = env_config_annotation(env_load_func, sig)
    if config_type is None:
        return env_args

    if "config" not in env_args:
        call_env_args = dict(env_args)
        call_env_args["config"] = coerce_config(
            config_type, prepare_base_env_config(config_type, {}, env_id=env_id)
        )
        return call_env_args

    config = env_args["config"]
    if config is None:
        raise TypeError("load_environment config must be a concrete EnvConfig object.")
    if isinstance(config, config_type):
        if config_type is EnvConfig:
            call_env_args = dict(env_args)
            call_env_args["config"] = coerce_config(
                config_type,
                prepare_base_env_config(config_type, config, env_id=env_id),
            )
            return call_env_args
        return env_args

    call_env_args = dict(env_args)
    call_env_args["config"] = coerce_config(
        config_type, prepare_base_env_config(config_type, config, env_id=env_id)
    )
    return call_env_args


def prepare_base_env_config(
    config_type: type[EnvConfig],
    config: object,
    *,
    env_id: str | None,
) -> object:
    if config_type is not EnvConfig or not env_id:
        return config
    data = explicit_config_data(config)
    taskset = data.get("taskset")
    taskset_data = config_table(taskset)
    if "id" not in taskset_data and "taskset_id" not in taskset_data:
        taskset_data["id"] = env_id
    data["taskset"] = taskset_data
    harness = data.get("harness")
    data["harness"] = config_table(harness)
    return data


def config_table(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return explicit_config_data(value)
    if not isinstance(value, Mapping):
        raise TypeError("EnvConfig child config must be a mapping or config object.")
    data: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("EnvConfig child config keys must be strings.")
        data[key] = item
    return data


def env_config_annotation(
    env_load_func: Callable[..., Environment],
    sig: inspect.Signature,
) -> type[EnvConfig] | None:
    if "config" not in sig.parameters:
        return None
    try:
        annotation = get_type_hints(env_load_func).get("config")
    except Exception:
        annotation = sig.parameters["config"].annotation
    return env_config_type(annotation)


def env_config_type(annotation: object) -> type[EnvConfig] | None:
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, type) and issubclass(annotation, EnvConfig):
        return annotation
    return None
