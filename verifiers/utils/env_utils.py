import importlib
import inspect
import logging
from types import ModuleType
from typing import Callable

from verifiers.envs.environment import Environment
from verifiers.utils.config_utils import MissingKeyError


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
            if hasattr(module, "load_taskset"):
                from verifiers.v1.compat import load_v1_environment_from_module

                env_instance = load_v1_environment_from_module(module, env_id, env_args)
                logger.info(f"Successfully loaded v1 environment '{env_id}'")
                return env_instance
            raise AttributeError(
                f"Module '{module.__name__}' does not expose load_environment."
            )

        from verifiers.v1.compat import (
            as_v0_environment,
            is_v1_load_environment,
            load_v1_environment_from_module,
        )

        if is_v1_load_environment(env_load_func, module):
            env_instance = load_v1_environment_from_module(module, env_id, env_args)
            logger.info(f"Successfully loaded v1 environment '{env_id}'")
            return env_instance

        sig = inspect.signature(env_load_func)
        provided_params = set(env_args.keys())
        default_params = set(sig.parameters.keys()) - provided_params

        if provided_params:
            provided_values = [f"{name}={env_args[name]}" for name in provided_params]
            logger.info(f"Using provided args: {', '.join(provided_values)}")
        if default_params:
            default_values = [
                f"{name}={sig.parameters[name].default!r}"
                for name in default_params
                if sig.parameters[name].default is not inspect.Parameter.empty
            ]
            if default_values:
                logger.info(f"Using default args: {', '.join(default_values)}")

        env_instance: Environment = as_v0_environment(
            env_load_func(**env_args), env_id=env_id, env_args=env_args
        )
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
