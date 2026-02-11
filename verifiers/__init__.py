__version__ = "0.1.10"

import importlib
import os
from typing import TYPE_CHECKING

# early imports to avoid circular dependencies
from .errors import *  # noqa # isort: skip
from .types import *  # noqa # isort: skip
from .decorators import (  # noqa # isort: skip
    cleanup,
    stop,
    teardown,
)
from .types import DatasetBuilder  # noqa # isort: skip
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
from .utils.config_utils import MissingKeyError, ensure_keys
from .utils.data_utils import (
    extract_boxed_answer,
    extract_hash_answer,
    load_example_dataset,
)
from .utils.env_utils import load_environment
from .utils.logging_utils import (
    log_level,
    print_prompt_completions_sample,
    quiet_verifiers,
    setup_logging,
)

# Setup default logging configuration
setup_logging(os.getenv("VF_LOG_LEVEL"))

__all__ = [
    "DatasetBuilder",
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
    "BrowserEnv",
    "OpenEnvEnv",
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
    "log_level",
    "quiet_verifiers",
    "load_environment",
    "print_prompt_completions_sample",
    "cleanup",
    "stop",
    "teardown",
    "ensure_keys",
    "MissingKeyError",
    "get_model",
    "get_model_and_tokenizer",
    "RLConfig",
    "RLTrainer",
    "GRPOTrainer",
    "GRPOConfig",
    "grpo_defaults",
    "lora_defaults",
]

_LAZY_IMPORTS = {
    "get_model": "verifiers_rl.rl.trainer.utils:get_model",
    "get_model_and_tokenizer": "verifiers_rl.rl.trainer.utils:get_model_and_tokenizer",
    "RLConfig": "verifiers_rl.rl.trainer:RLConfig",
    "RLTrainer": "verifiers_rl.rl.trainer:RLTrainer",
    "GRPOTrainer": "verifiers_rl.rl.trainer:GRPOTrainer",
    "GRPOConfig": "verifiers_rl.rl.trainer:GRPOConfig",
    "grpo_defaults": "verifiers_rl.rl.trainer:grpo_defaults",
    "lora_defaults": "verifiers_rl.rl.trainer:lora_defaults",
    "MathRubric": "verifiers.rubrics.math_rubric:MathRubric",
    "SandboxEnv": "verifiers.envs.sandbox_env:SandboxEnv",
    "PythonEnv": "verifiers.envs.python_env:PythonEnv",
    "GymEnv": "verifiers.envs.experimental.gym_env:GymEnv",
    "CliAgentEnv": "verifiers.envs.experimental.cli_agent_env:CliAgentEnv",
    "HarborEnv": "verifiers.envs.experimental.harbor_env:HarborEnv",
    "MCPEnv": "verifiers.envs.experimental.mcp_env:MCPEnv",
    "ReasoningGymEnv": "verifiers.envs.integrations.reasoninggym_env:ReasoningGymEnv",
    "TextArenaEnv": "verifiers.envs.integrations.textarena_env:TextArenaEnv",
    "BrowserEnv": "verifiers.envs.integrations.browser_env:BrowserEnv",
    "OpenEnvEnv": "verifiers.envs.integrations.openenv_env:OpenEnvEnv",
}


def __getattr__(name: str):
    try:
        module, attr = _LAZY_IMPORTS[name].split(":")
        return getattr(importlib.import_module(module), attr)
    except KeyError:
        raise AttributeError(f"module 'verifiers' has no attribute '{name}'")
    except ModuleNotFoundError as e:
        rl_names = {
            "get_model",
            "get_model_and_tokenizer",
            "RLConfig",
            "RLTrainer",
            "GRPOTrainer",
            "GRPOConfig",
            "grpo_defaults",
            "lora_defaults",
        }
        if name in rl_names:
            raise AttributeError(
                f"To use verifiers.{name}, install as `verifiers-rl`."
            ) from e
        raise AttributeError(
            f"To use verifiers.{name}, install as `verifiers[all]`. "
        ) from e


if TYPE_CHECKING:
    from typing import Any

    from .envs.experimental.cli_agent_env import CliAgentEnv  # noqa: F401
    from .envs.experimental.gym_env import GymEnv  # noqa: F401
    from .envs.experimental.harbor_env import HarborEnv  # noqa: F401
    from .envs.experimental.mcp_env import MCPEnv  # noqa: F401
    from .envs.integrations.browser_env import BrowserEnv  # noqa: F401
    from .envs.integrations.openenv_env import OpenEnvEnv  # noqa: F401
    from .envs.integrations.reasoninggym_env import ReasoningGymEnv  # noqa: F401
    from .envs.integrations.textarena_env import TextArenaEnv  # noqa: F401
    from .envs.python_env import PythonEnv  # noqa: F401
    from .envs.sandbox_env import SandboxEnv  # noqa: F401
    from .rubrics.math_rubric import MathRubric  # noqa: F401

    # Optional verifiers-rl exports. Keep type-checking clean when extra is absent.
    RLConfig: Any
    RLTrainer: Any
    GRPOTrainer: Any
    GRPOConfig: Any
    grpo_defaults: Any
    lora_defaults: Any
    get_model: Any
    get_model_and_tokenizer: Any
