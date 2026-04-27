__version__ = "0.1.13.dev7"

import importlib
import os
from typing import TYPE_CHECKING

# early imports to avoid circular dependencies
from .errors import *  # noqa # isort: skip
from .types import *  # noqa # isort: skip
from .decorators import (  # noqa # isort: skip
    advantage,
    cleanup,
    metric,
    render,
    reward,
    stop,
    teardown,
)
from .types import DatasetBuilder  # noqa # isort: skip
from .parsers.parser import Parser  # noqa # isort: skip
from .rubrics.rubric import Rubric  # noqa # isort: skip

# main imports
from .parsers.maybe_think_parser import MaybeThinkParser
from .parsers.think_parser import ThinkParser
from .parsers.xml_parser import XMLParser
from .rubrics.rubric_group import RubricGroup
from .utils.config_utils import MissingKeyError, ensure_keys
from .utils.data_utils import (
    extract_boxed_answer,
    extract_hash_answer,
    load_example_dataset,
)
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
    "Env",
    "Binding",
    "BindingContext",
    "Channel",
    "ChannelConfig",
    "ChannelContext",
    "ChannelMap",
    "CliConfig",
    "CliMetrics",
    "CliPaths",
    "Task",
    "Taskset",
    "Harness",
    "EndpointConfig",
    "SandboxBashTool",
    "SandboxEditTool",
    "SandboxPythonTool",
    "RunConfig",
    "SandboxConfig",
    "SandboxSpec",
    "SandboxSeed",
    "SandboxTool",
    "SandboxRuntime",
    "SandboxScoring",
    "SandboxSetup",
    "SandboxTimeouts",
    "Resources",
    "CallableTool",
    "MCPTool",
    "ResourceBinding",
    "Toolset",
    "StateBinding",
    "TaskBinding",
    "User",
    "Environment",
    "MultiTurnEnv",
    "SingleTurnEnv",
    "PythonEnv",
    "SandboxEnv",
    "StatefulToolEnv",
    "ToolEnv",
    "EnvGroup",
    "Client",
    "AnthropicMessagesClient",
    "OpenAIChatCompletionsClient",
    "OpenAIChatCompletionsTokenClient",
    "OpenAICompletionsClient",
    "extract_boxed_answer",
    "extract_hash_answer",
    "load_example_dataset",
    "setup_logging",
    "log_level",
    "quiet_verifiers",
    "load_environment",
    "print_prompt_completions_sample",
    "cleanup",
    "render",
    "metric",
    "reward",
    "advantage",
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
    "Client": "verifiers.clients.client:Client",
    "AnthropicMessagesClient": (
        "verifiers.clients.anthropic_messages_client:AnthropicMessagesClient"
    ),
    "OpenAIChatCompletionsClient": (
        "verifiers.clients.openai_chat_completions_client:OpenAIChatCompletionsClient"
    ),
    "OpenAIChatCompletionsTokenClient": (
        "verifiers.clients.openai_chat_completions_token_client:OpenAIChatCompletionsTokenClient"
    ),
    "OpenAICompletionsClient": (
        "verifiers.clients.openai_completions_client:OpenAICompletionsClient"
    ),
    "Environment": "verifiers.envs.environment:Environment",
    "MultiTurnEnv": "verifiers.envs.multiturn_env:MultiTurnEnv",
    "SingleTurnEnv": "verifiers.envs.singleturn_env:SingleTurnEnv",
    "StatefulToolEnv": "verifiers.envs.stateful_tool_env:StatefulToolEnv",
    "ToolEnv": "verifiers.envs.tool_env:ToolEnv",
    "EnvGroup": "verifiers.envs.env_group:EnvGroup",
    "JudgeRubric": "verifiers.rubrics.judge_rubric:JudgeRubric",
    "load_environment": "verifiers.utils.env_utils:load_environment",
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
    "Env": "verifiers.envs.experimental.env:Env",
    "Binding": "verifiers.envs.experimental.binding:Binding",
    "BindingContext": "verifiers.envs.experimental.binding:BindingContext",
    "Channel": "verifiers.envs.experimental.channels:Channel",
    "ChannelConfig": "verifiers.envs.experimental.channels:ChannelConfig",
    "ChannelContext": "verifiers.envs.experimental.channels:ChannelContext",
    "ChannelMap": "verifiers.envs.experimental.channels:ChannelMap",
    "CliConfig": "verifiers.envs.experimental.configs:CliConfig",
    "CliMetrics": "verifiers.envs.experimental.configs:CliMetrics",
    "CliPaths": "verifiers.envs.experimental.configs:CliPaths",
    "Task": "verifiers.envs.experimental.task:Task",
    "Taskset": "verifiers.envs.experimental.taskset:Taskset",
    "Harness": "verifiers.envs.experimental.harness:Harness",
    "EndpointConfig": "verifiers.envs.experimental.configs:EndpointConfig",
    "SandboxBashTool": "verifiers.envs.experimental.modules.tools:SandboxBashTool",
    "SandboxEditTool": "verifiers.envs.experimental.modules.tools:SandboxEditTool",
    "SandboxPythonTool": "verifiers.envs.experimental.modules.tools:SandboxPythonTool",
    "RunConfig": "verifiers.envs.experimental.configs:RunConfig",
    "SandboxConfig": "verifiers.envs.experimental.configs:SandboxConfig",
    "SandboxSpec": "verifiers.envs.experimental.channels:SandboxSpec",
    "SandboxSeed": "verifiers.envs.experimental.channels:SandboxSeed",
    "SandboxTool": "verifiers.envs.experimental.modules.tools:SandboxTool",
    "SandboxRuntime": "verifiers.envs.experimental.configs:SandboxRuntime",
    "SandboxScoring": "verifiers.envs.experimental.configs:SandboxScoring",
    "SandboxSetup": "verifiers.envs.experimental.configs:SandboxSetup",
    "SandboxTimeouts": "verifiers.envs.experimental.channels:SandboxTimeouts",
    "Resources": "verifiers.envs.experimental.resources:Resources",
    "CallableTool": "verifiers.envs.experimental.toolset:CallableTool",
    "MCPTool": "verifiers.envs.experimental.toolset:MCPTool",
    "ResourceBinding": "verifiers.envs.experimental.binding:ResourceBinding",
    "StateBinding": "verifiers.envs.experimental.binding:StateBinding",
    "TaskBinding": "verifiers.envs.experimental.binding:TaskBinding",
    "Toolset": "verifiers.envs.experimental.toolset:Toolset",
    "User": "verifiers.envs.experimental.channels:User",
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

    from .clients.anthropic_messages_client import AnthropicMessagesClient  # noqa: F401
    from .clients.client import Client  # noqa: F401
    from .clients.openai_chat_completions_client import (  # noqa: F401
        OpenAIChatCompletionsClient,
    )
    from .clients.openai_chat_completions_token_client import (  # noqa: F401
        OpenAIChatCompletionsTokenClient,
    )
    from .clients.openai_completions_client import OpenAICompletionsClient  # noqa: F401
    from .envs.env_group import EnvGroup  # noqa: F401
    from .envs.environment import Environment  # noqa: F401
    from .envs.experimental.cli_agent_env import CliAgentEnv  # noqa: F401
    from .envs.experimental.gym_env import GymEnv  # noqa: F401
    from .envs.experimental.harbor_env import HarborEnv  # noqa: F401
    from .envs.experimental.mcp_env import MCPEnv  # noqa: F401
    from .envs.integrations.browser_env import BrowserEnv  # noqa: F401
    from .envs.integrations.openenv_env import OpenEnvEnv  # noqa: F401
    from .envs.integrations.reasoninggym_env import ReasoningGymEnv  # noqa: F401
    from .envs.integrations.textarena_env import TextArenaEnv  # noqa: F401
    from .envs.experimental.env import Env  # noqa: F401
    from .envs.experimental.binding import (  # noqa: F401
        Binding,
        BindingContext,
        ResourceBinding,
        StateBinding,
        TaskBinding,
    )
    from .envs.experimental.channels import (  # noqa: F401
        Channel,
        ChannelConfig,
        ChannelContext,
        ChannelMap,
    )
    from .envs.experimental.configs import (  # noqa: F401
        CliConfig,
        CliMetrics,
        CliPaths,
        EndpointConfig,
        RunConfig,
        SandboxConfig,
        SandboxRuntime,
        SandboxScoring,
        SandboxSetup,
    )
    from .envs.experimental.harness import Harness  # noqa: F401
    from .envs.experimental.modules.tools import (
        SandboxBashTool,
        SandboxEditTool,
        SandboxPythonTool,
        SandboxTool,
    )  # noqa: F401
    from .envs.experimental.resources import Resources  # noqa: F401
    from .envs.experimental.channels import SandboxSeed, SandboxSpec, SandboxTimeouts  # noqa: F401
    from .envs.experimental.task import Task  # noqa: F401
    from .envs.experimental.taskset import Taskset  # noqa: F401
    from .envs.experimental.channels import User  # noqa: F401
    from .envs.experimental.toolset import CallableTool, MCPTool, Toolset  # noqa: F401
    from .envs.multiturn_env import MultiTurnEnv  # noqa: F401
    from .envs.python_env import PythonEnv  # noqa: F401
    from .envs.sandbox_env import SandboxEnv  # noqa: F401
    from .envs.singleturn_env import SingleTurnEnv  # noqa: F401
    from .envs.stateful_tool_env import StatefulToolEnv  # noqa: F401
    from .envs.tool_env import ToolEnv  # noqa: F401
    from .rubrics.judge_rubric import JudgeRubric  # noqa: F401
    from .rubrics.math_rubric import MathRubric  # noqa: F401
    from .utils.env_utils import load_environment  # noqa: F401

    # Optional verifiers-rl exports. Keep type-checking clean when extra is absent.
    RLConfig: Any
    RLTrainer: Any
    GRPOTrainer: Any
    GRPOConfig: Any
    grpo_defaults: Any
    lora_defaults: Any
    get_model: Any
    get_model_and_tokenizer: Any
