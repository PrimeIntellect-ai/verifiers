import importlib
import inspect
import logging
from typing import Callable, cast

from verifiers.envs.environment import Environment


def load_environment(
    env_id: str,
    tools: list[Callable] | list[str] | None = None,
    **env_args,
) -> Environment:
    logger = logging.getLogger("verifiers.utils.env_utils")
    logger.info(f"Loading environment: {env_id}")

    # Phase 1: Pre-process tools parameter
    tool_names_to_resolve = None

    if tools is not None:
        # Check if tools is actually a list (not a string or other type)
        if not isinstance(tools, list):
            raise TypeError(
                f"tools must be a list, got {type(tools).__name__}. "
                f"If passing tool names, use tools=['tool_name'] not tools='tool_name'"
            )

        if tools:
            # Check for mixed types (both Callable and str in same list)
            first_is_callable = callable(tools[0])
            first_is_str = isinstance(tools[0], str)

            if not first_is_callable and not first_is_str:
                raise TypeError(
                    f"tools must be list of Callable or list of str, "
                    f"got list containing {type(tools[0]).__name__}"
                )

            # Verify all tools are same type AND are valid types (callable or str)
            for i, tool in enumerate(tools):
                is_callable = callable(tool)
                is_str = isinstance(tool, str)

                # Check if element is neither callable nor string
                if not is_callable and not is_str:
                    raise TypeError(
                        f"tools list elements must be Callable or str, "
                        f"got {type(tool).__name__} at index {i}"
                    )

                # Check for mixed types
                if is_callable and first_is_str:
                    raise TypeError(
                        f"tools must be all Callable or all str, got mixed types "
                        f"(tool[0] is {type(tools[0]).__name__}, tool[{i}] is {type(tool).__name__})"
                    )
                if is_str and first_is_callable:
                    raise TypeError(
                        f"tools must be all Callable or all str, got mixed types "
                        f"(tool[0] is {type(tools[0]).__name__}, tool[{i}] is {type(tool).__name__})"
                    )

            if first_is_str:
                # String list: store for later resolution AFTER module import
                tool_names_to_resolve = cast(list[str], tools)
                logger.info(f"Will resolve tools after import: {tool_names_to_resolve}")
            else:
                # Callable list: pass through immediately
                env_args["tools"] = tools
                logger.info(f"Using callable tools directly: {len(tools)} tools")
        else:
            # Empty list: explicitly set to no tools
            env_args["tools"] = []
            logger.info("Using empty tool list")

    # Phase 2: Import environment module FIRST
    # This triggers @register_tool decorators, populating the registry
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

        # Phase 3: NOW resolve string tools (registry is populated!)
        if tool_names_to_resolve is not None:
            from verifiers.utils.tool_registry import get_tools

            logger.info(f"Resolving tools from registry: {tool_names_to_resolve}")
            try:
                tools = get_tools(env_id, tool_names_to_resolve)
                env_args["tools"] = tools
                logger.info(f"Successfully resolved {len(tools)} tools")
            except KeyError as e:
                logger.error(
                    f"Failed to resolve tools for env '{env_id}': {str(e)}\n"
                    f"Note: Tools must be registered with @register_tool decorator "
                    f"in the environment module before load_environment() is called."
                )
                raise
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
