__version__ = "0.1.9.post0"

import importlib
import logging
import os
import sys
from typing import Optional, TYPE_CHECKING


# Setup default logging configuration
def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """
    Setup basic logging configuration for the verifiers package.

    Args:
        level: The logging level to use. Defaults to "INFO".
        log_format: Custom log format string. If None, uses default format.
        date_format: Custom date format string. If None, uses default format.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Create a StreamHandler that writes to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    # Get the root logger for the verifiers package
    logger = logging.getLogger("verifiers")
    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()
    # Add a new handler with desired log level
    logger.setLevel(level.upper())
    logger.addHandler(handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False


setup_logging(os.getenv("VF_LOG_LEVEL", "INFO"))

__all__ = [
    "Parser",
    "ThinkParser",
    "MaybeThinkParser",
    "XMLParser",
    "Rubric",
    "JudgeRubric",
    "RubricGroup",
    "MathRubric",
    "TextArenaEnv",
    "ReasoningGymEnv",
    "GymEnv",
    "CliAgentEnv",
    "HarborEnv",
    "MCPEnv",
    "Environment",
    "MultiTurnEnv",
    "SingleTurnEnv",
    "PythonEnv",
    "SandboxEnv",
    "StatefulToolEnv",
    "ToolEnv",
    "EnvGroup",
    "extract_boxed_answer",
    "extract_hash_answer",
    "load_example_dataset",
    "setup_logging",
    "load_environment",
    "print_prompt_completions_sample",
    "get_model",
    "get_model_and_tokenizer",
    "RLTrainer",
    "RLConfig",
    "GRPOTrainer",
    "GRPOConfig",
    "grpo_defaults",
    "lora_defaults",
    "cleanup",
    "stop",
    "teardown",
    "Error",
    "ModelError",
    "EmptyModelResponseError",
    "OverlongPromptError",
    "ToolError",
    "ToolParseError",
    "ToolCallError",
    "InfraError",
    "SandboxError",
]

_LAZY_IMPORTS = {
    "Parser": "verifiers.parsers.parser:Parser",
    "ThinkParser": "verifiers.parsers.think_parser:ThinkParser",
    "MaybeThinkParser": "verifiers.parsers.maybe_think_parser:MaybeThinkParser",
    "XMLParser": "verifiers.parsers.xml_parser:XMLParser",
    "Rubric": "verifiers.rubrics.rubric:Rubric",
    "JudgeRubric": "verifiers.rubrics.judge_rubric:JudgeRubric",
    "RubricGroup": "verifiers.rubrics.rubric_group:RubricGroup",
    "MathRubric": "verifiers.rubrics.math_rubric:MathRubric",
    "TextArenaEnv": "verifiers.envs.integrations.textarena_env:TextArenaEnv",
    "ReasoningGymEnv": "verifiers.envs.integrations.reasoninggym_env:ReasoningGymEnv",
    "GymEnv": "verifiers.envs.experimental.gym_env:GymEnv",
    "CliAgentEnv": "verifiers.envs.experimental.cli_agent_env:CliAgentEnv",
    "HarborEnv": "verifiers.envs.experimental.harbor_env:HarborEnv",
    "MCPEnv": "verifiers.envs.experimental.mcp_env:MCPEnv",
    "Environment": "verifiers.envs.environment:Environment",
    "MultiTurnEnv": "verifiers.envs.multiturn_env:MultiTurnEnv",
    "SingleTurnEnv": "verifiers.envs.singleturn_env:SingleTurnEnv",
    "PythonEnv": "verifiers.envs.python_env:PythonEnv",
    "SandboxEnv": "verifiers.envs.sandbox_env:SandboxEnv",
    "StatefulToolEnv": "verifiers.envs.stateful_tool_env:StatefulToolEnv",
    "ToolEnv": "verifiers.envs.tool_env:ToolEnv",
    "EnvGroup": "verifiers.envs.env_group:EnvGroup",
    "extract_boxed_answer": "verifiers.utils.data_utils:extract_boxed_answer",
    "extract_hash_answer": "verifiers.utils.data_utils:extract_hash_answer",
    "load_example_dataset": "verifiers.utils.data_utils:load_example_dataset",
    "load_environment": "verifiers.utils.env_utils:load_environment",
    "print_prompt_completions_sample": (
        "verifiers.utils.logging_utils:print_prompt_completions_sample"
    ),
    "get_model": "verifiers.rl.trainer.utils:get_model",
    "get_model_and_tokenizer": "verifiers.rl.trainer.utils:get_model_and_tokenizer",
    "RLTrainer": "verifiers.rl.trainer:RLTrainer",
    "RLConfig": "verifiers.rl.trainer:RLConfig",
    "GRPOTrainer": "verifiers.rl.trainer:GRPOTrainer",
    "GRPOConfig": "verifiers.rl.trainer:GRPOConfig",
    "grpo_defaults": "verifiers.rl.trainer:grpo_defaults",
    "lora_defaults": "verifiers.rl.trainer:lora_defaults",
    "cleanup": "verifiers.decorators:cleanup",
    "stop": "verifiers.decorators:stop",
    "teardown": "verifiers.decorators:teardown",
    "Error": "verifiers.errors:Error",
    "ModelError": "verifiers.errors:ModelError",
    "EmptyModelResponseError": "verifiers.errors:EmptyModelResponseError",
    "OverlongPromptError": "verifiers.errors:OverlongPromptError",
    "ToolError": "verifiers.errors:ToolError",
    "ToolParseError": "verifiers.errors:ToolParseError",
    "ToolCallError": "verifiers.errors:ToolCallError",
    "InfraError": "verifiers.errors:InfraError",
    "SandboxError": "verifiers.errors:SandboxError",
}


def __getattr__(name: str):
    target = _LAZY_IMPORTS.get(name)
    if target is not None:
        module, attr = target.split(":")
        try:
            return getattr(importlib.import_module(module), attr)
        except ModuleNotFoundError as e:
            # warn that accessed var needs [all] to be installed
            raise AttributeError(
                f"To use verifiers.{name}, install as `verifiers[all]`. "
            ) from e

    for module_name in ("verifiers.errors", "verifiers.types"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        if hasattr(module, name):
            return getattr(module, name)

    raise AttributeError(f"module 'verifiers' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__) | set(_LAZY_IMPORTS.keys()))


if TYPE_CHECKING:
    from .envs.experimental.cli_agent_env import CliAgentEnv  # noqa: F401
    from .envs.experimental.gym_env import GymEnv  # noqa: F401
    from .envs.experimental.harbor_env import HarborEnv  # noqa: F401
    from .envs.experimental.mcp_env import MCPEnv  # noqa: F401
    from .envs.integrations.reasoninggym_env import ReasoningGymEnv  # noqa: F401
    from .envs.integrations.textarena_env import TextArenaEnv  # noqa: F401
    from .envs.python_env import PythonEnv  # noqa: F401
    from .envs.sandbox_env import SandboxEnv  # noqa: F401
    from .rl.trainer import (  # noqa: F401
        GRPOConfig,
        GRPOTrainer,
        RLConfig,
        RLTrainer,
        grpo_defaults,
        lora_defaults,
    )
    from .rl.trainer.utils import (  # noqa: F401
        get_model,
        get_model_and_tokenizer,
    )
    from .rubrics.math_rubric import MathRubric  # noqa: F401
