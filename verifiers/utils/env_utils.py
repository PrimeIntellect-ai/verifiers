import importlib
import inspect
import logging
from typing import Callable

from verifiers.envs.environment import Environment


def load_environment(env_id: str, **env_args) -> Environment:
    logger = logging.getLogger("verifiers.utils.env_utils")
    logger.info(f"Loading environment: {env_id}")

    module_name = env_id.replace("-", "_").split("/")[-1]
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

        env_instance: Environment = env_load_func(**env_args)
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
    except Exception as e:
        logger.error(
            f"Failed to load environment {env_id} with args {env_args}: {str(e)}"
        )
        raise RuntimeError(f"Failed to load environment '{env_id}': {str(e)}") from e
