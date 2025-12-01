__version__ = "0.1.8.post1"

import importlib
import logging
import os
import sys
from typing import TYPE_CHECKING, Optional

# early imports to avoid circular dependencies
from .types import *  # noqa # isort: skip
from .utils.decorators import (  # noqa # isort: skip
    cleanup,
    stop,
    teardown,
)
from .parsers.parser import Parser  # noqa # isort: skip
from .rubrics.rubric import Rubric  # noqa # isort: skip
from .envs.environment import Environment  # noqa # isort: skip
from .envs.multiturn_env import MultiTurnEnv  # noqa # isort: skip
from .envs.tool_env import ToolEnv  # noqa # isort: skip

# main imports
from .envs.env_group import EnvGroup
from .envs.singleturn_env import SingleTurnEnv
from .envs.stateful_tool_env import StatefulToolEnv
from .parsers.maybe_think_parser import MaybeThinkParser
from .parsers.think_parser import ThinkParser
from .parsers.xml_parser import XMLParser
from .rubrics.judge_rubric import JudgeRubric
from .rubrics.rubric_group import RubricGroup
from .rubrics.tool_rubric import ToolRubric
from .utils.data_utils import (
    extract_boxed_answer,
    extract_hash_answer,
    load_example_dataset,
)
from .utils.env_utils import load_environment
from .utils.logging_utils import print_prompt_completions_sample


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
    "ToolRubric",
    "MathRubric",
    "TextArenaEnv",
    "ReasoningGymEnv",
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
]

_LAZY_IMPORTS = {
    "get_model": "verifiers.rl.trainer.utils:get_model",
    "get_model_and_tokenizer": "verifiers.rl.trainer.utils:get_model_and_tokenizer",
    "RLConfig": "verifiers.rl.trainer:RLConfig",
    "RLTrainer": "verifiers.rl.trainer:RLTrainer",
    "GRPOTrainer": "verifiers.rl.trainer:GRPOTrainer",
    "GRPOConfig": "verifiers.rl.trainer:GRPOConfig",
    "grpo_defaults": "verifiers.rl.trainer:grpo_defaults",
    "lora_defaults": "verifiers.rl.trainer:lora_defaults",
    "MathRubric": "verifiers.rubrics.math_rubric:MathRubric",
    "SandboxEnv": "verifiers.envs.sandbox_env:SandboxEnv",
    "PythonEnv": "verifiers.envs.python_env:PythonEnv",
    "ReasoningGymEnv": "verifiers.envs.reasoninggym_env:ReasoningGymEnv",
    "TextArenaEnv": "verifiers.envs.textarena_env:TextArenaEnv",
}


def __getattr__(name: str):
    try:
        module, attr = _LAZY_IMPORTS[name].split(":")
        return getattr(importlib.import_module(module), attr)
    except KeyError:
        raise AttributeError(f"module 'verifiers' has no attribute '{name}'")
    except ModuleNotFoundError as e:
        # warn that accessed var needs [all] to be installed
        raise AttributeError(
            f"To use verifiers.{name}, install as `verifiers[all]`. "
        ) from e


if TYPE_CHECKING:
    from .envs.python_env import PythonEnv  # noqa: F401
    from .envs.reasoninggym_env import ReasoningGymEnv  # noqa: F401
    from .envs.sandbox_env import SandboxEnv  # noqa: F401
    from .envs.textarena_env import TextArenaEnv  # noqa: F401
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
