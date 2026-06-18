from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("verifiers")
except PackageNotFoundError:  # source tree without install metadata
    __version__ = "0.0.0+unknown"

import importlib
import os
from typing import TYPE_CHECKING, Literal, TypeAlias

# early imports to avoid circular dependencies
from .errors import (  # noqa # isort: skip
    BrowserSandboxError,
    EmptyModelResponseError,
    Error,
    InfraError,
    InvalidModelResponseError,
    ModelError,
    OverlongPromptError,
    SandboxDeleteError,
    SandboxError,
    ToolCallError,
    ToolError,
    ToolParseError,
    TunnelError,
)
from .types import *  # noqa # isort: skip
from .decorators import (  # noqa # isort: skip
    advantage,
    cleanup,
    metric,
    reward,
    setup,
    stop,
    teardown,
    update,
)
from .types import DatasetBuilder, EndpointConfig, Endpoints, State  # noqa # isort: skip
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

TaskSplit: TypeAlias = Literal["train", "eval"]

# Setup default logging configuration
setup_logging(os.getenv("VF_LOG_LEVEL"))

__all__ = [
    "ArtifactConfig",
    "Artifacts",
    "ArtifactsConfig",
    "DatasetBuilder",
    "State",
    "Error",
    "ModelError",
    "InvalidModelResponseError",
    "EmptyModelResponseError",
    "OverlongPromptError",
    "ToolError",
    "ToolParseError",
    "ToolCallError",
    "InfraError",
    "TunnelError",
    "SandboxError",
    "SandboxDeleteError",
    "BrowserSandboxError",
    "BindingsConfig",
    "CallableConfig",
    "Config",
    "ConfigData",
    "Handler",
    "JsonData",
    "Objects",
    "ObjectsConfig",
    "ProgramConfig",
    "ProgramValue",
    "PromptInput",
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
    "EnvConfig",
    "Endpoint",
    "EndpointConfig",
    "Endpoints",
    "Task",
    "TaskSplit",
    "Tasks",
    "Taskset",
    "TasksetConfig",
    "Harness",
    "HarnessConfig",
    "MCPTool",
    "MCPToolConfig",
    "ModelConfig",
    "SandboxConfig",
    "SystemPrompt",
    "SystemPromptConfig",
    "SystemPromptStrategy",
    "Toolset",
    "ToolLike",
    "ToolsetConfig",
    "Toolsets",
    "TrajectoryVisibility",
    "User",
    "UserConfig",
    "VisibilityConfig",
    "SignalConfig",
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
    "OpenAICompletionsClient",
    "OpenAIResponsesClient",
    "RendererClient",
    "extract_boxed_answer",
    "extract_hash_answer",
    "load_example_dataset",
    "setup_logging",
    "log_level",
    "quiet_verifiers",
    "load_environment",
    "load_harness",
    "load_taskset",
    "print_prompt_completions_sample",
    "get_messages",
    "cleanup",
    "metric",
    "reward",
    "advantage",
    "setup",
    "stop",
    "teardown",
    "update",
    "add_metric",
    "add_reward",
    "add_advantage",
    "build_signals",
    "collect_signals",
    "score_group",
    "score_rollout",
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
    "RendererClient": ("verifiers.clients.renderer_client:RendererClient"),
    "OpenAICompletionsClient": (
        "verifiers.clients.openai_completions_client:OpenAICompletionsClient"
    ),
    "OpenAIResponsesClient": (
        "verifiers.clients.openai_responses_client:OpenAIResponsesClient"
    ),
    "Environment": "verifiers.envs.environment:Environment",
    "MultiTurnEnv": "verifiers.envs.multiturn_env:MultiTurnEnv",
    "SingleTurnEnv": "verifiers.envs.singleturn_env:SingleTurnEnv",
    "StatefulToolEnv": "verifiers.envs.stateful_tool_env:StatefulToolEnv",
    "ToolEnv": "verifiers.envs.tool_env:ToolEnv",
    "EnvGroup": "verifiers.envs.env_group:EnvGroup",
    "JudgeRubric": "verifiers.rubrics.judge_rubric:JudgeRubric",
    "load_environment": "verifiers.utils.env_utils:load_environment",
    "load_harness": "verifiers.utils.env_utils:load_harness",
    "load_taskset": "verifiers.utils.env_utils:load_taskset",
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
    "Config": "verifiers.v1:Config",
    "CallableConfig": "verifiers.v1:CallableConfig",
    "BindingsConfig": "verifiers.v1:BindingsConfig",
    "ArtifactConfig": "verifiers.v1:ArtifactConfig",
    "Artifacts": "verifiers.v1:Artifacts",
    "ArtifactsConfig": "verifiers.v1:ArtifactsConfig",
    "Env": "verifiers.v1:Env",
    "EnvConfig": "verifiers.v1:EnvConfig",
    "Endpoint": "verifiers.v1:Endpoint",
    "EndpointConfig": "verifiers.v1:EndpointConfig",
    "ConfigData": "verifiers.v1:ConfigData",
    "Handler": "verifiers.v1:Handler",
    "JsonData": "verifiers.v1:JsonData",
    "Objects": "verifiers.v1:Objects",
    "ObjectsConfig": "verifiers.v1:ObjectsConfig",
    "Task": "verifiers.v1:Task",
    "TaskSplit": "verifiers.v1:TaskSplit",
    "Tasks": "verifiers.v1:Tasks",
    "Taskset": "verifiers.v1:Taskset",
    "TasksetConfig": "verifiers.v1:TasksetConfig",
    "Harness": "verifiers.v1:Harness",
    "HarnessConfig": "verifiers.v1:HarnessConfig",
    "ProgramConfig": "verifiers.v1:ProgramConfig",
    "ProgramValue": "verifiers.v1:ProgramValue",
    "PromptInput": "verifiers.v1:PromptInput",
    "MCPTool": "verifiers.v1:MCPTool",
    "MCPToolConfig": "verifiers.v1:MCPToolConfig",
    "ModelConfig": "verifiers.v1:ModelConfig",
    "SandboxConfig": "verifiers.v1:SandboxConfig",
    "SignalConfig": "verifiers.v1:SignalConfig",
    "SystemPrompt": "verifiers.v1:SystemPrompt",
    "SystemPromptConfig": "verifiers.v1:SystemPromptConfig",
    "SystemPromptStrategy": "verifiers.v1:SystemPromptStrategy",
    "ToolLike": "verifiers.v1:ToolLike",
    "Toolset": "verifiers.v1:Toolset",
    "ToolsetConfig": "verifiers.v1:ToolsetConfig",
    "Toolsets": "verifiers.v1:Toolsets",
    "TrajectoryVisibility": "verifiers.v1:TrajectoryVisibility",
    "User": "verifiers.v1:User",
    "UserConfig": "verifiers.v1:UserConfig",
    "VisibilityConfig": "verifiers.v1:VisibilityConfig",
    "get_messages": "verifiers.v1:get_messages",
    "add_metric": "verifiers.v1:add_metric",
    "add_reward": "verifiers.v1:add_reward",
    "add_advantage": "verifiers.v1:add_advantage",
    "build_signals": "verifiers.v1:build_signals",
    "collect_signals": "verifiers.v1:collect_signals",
    "score_group": "verifiers.v1:score_group",
    "score_rollout": "verifiers.v1:score_rollout",
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
        if name == "RendererClient":
            raise AttributeError(
                "To use verifiers.RendererClient, install as `verifiers[renderers]`."
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
    from .clients.openai_completions_client import OpenAICompletionsClient  # noqa: F401
    from .clients.openai_responses_client import OpenAIResponsesClient  # noqa: F401
    from .clients.renderer_client import RendererClient  # noqa: F401
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
    from .envs.multiturn_env import MultiTurnEnv  # noqa: F401
    from .envs.python_env import PythonEnv  # noqa: F401
    from .envs.sandbox_env import SandboxEnv  # noqa: F401
    from .envs.singleturn_env import SingleTurnEnv  # noqa: F401
    from .envs.stateful_tool_env import StatefulToolEnv  # noqa: F401
    from .envs.tool_env import ToolEnv  # noqa: F401
    from .rubrics.judge_rubric import JudgeRubric  # noqa: F401
    from .rubrics.math_rubric import MathRubric  # noqa: F401
    from .utils.env_utils import (  # noqa: F401
        load_environment,
        load_harness,
        load_taskset,
    )
    from .v1 import (  # noqa: F401
        ArtifactConfig,
        Artifacts,
        ArtifactsConfig,
        BindingsConfig,
        CallableConfig,
        Config,
        ConfigData,
        Env,
        EnvConfig,
        Endpoint,
        EndpointConfig,
        Handler,
        Harness,
        HarnessConfig,
        JsonData,
        MCPTool,
        MCPToolConfig,
        ModelConfig,
        Objects,
        ObjectsConfig,
        ProgramConfig,
        ProgramValue,
        PromptInput,
        SandboxConfig,
        SignalConfig,
        SystemPrompt,
        SystemPromptConfig,
        SystemPromptStrategy,
        Task,
        Tasks,
        Taskset,
        TasksetConfig,
        ToolLike,
        Toolset,
        ToolsetConfig,
        Toolsets,
        TrajectoryVisibility,
        User,
        UserConfig,
        VisibilityConfig,
        add_advantage,
        add_metric,
        add_reward,
        build_signals,
        collect_signals,
        get_messages,
        score_group,
        score_rollout,
    )

    # Optional verifiers-rl exports. Keep type-checking clean when extra is absent.
    RLConfig: Any
    RLTrainer: Any
    GRPOTrainer: Any
    GRPOConfig: Any
    grpo_defaults: Any
    lora_defaults: Any
    get_model: Any
    get_model_and_tokenizer: Any
