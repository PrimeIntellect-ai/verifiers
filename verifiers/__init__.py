__version__ = "0.1.11"

import importlib
import os
from typing import TYPE_CHECKING

from .errors import *  # noqa: F401,F403
from .utils.logging_utils import setup_logging as _setup_logging

_setup_logging(os.getenv("VF_LOG_LEVEL"))

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

_TYPE_EXPORTS = {
    "AssistantMessage",
    "BaseRolloutInput",
    "ClientConfig",
    "ClientType",
    "ContentPart",
    "CustomBaseModel",
    "DatasetBuilder",
    "Endpoint",
    "EndpointClientConfig",
    "Endpoints",
    "ErrorInfo",
    "EvalConfig",
    "EvalRunConfig",
    "FinishReason",
    "GenerateMetadata",
    "GenerateOutputs",
    "GenericContentPart",
    "GroupRewardFunc",
    "ImageUrlContentPart",
    "ImageUrlSource",
    "IndividualRewardFunc",
    "Info",
    "InputAudioContentPart",
    "InputAudioSource",
    "JsonPrimitive",
    "LogCallback",
    "Message",
    "MessageContent",
    "Messages",
    "MessageType",
    "ProgressCallback",
    "Response",
    "ResponseMessage",
    "ResponseTokens",
    "RewardFunc",
    "RolloutInput",
    "RolloutOutput",
    "RolloutScore",
    "RolloutScores",
    "RolloutTiming",
    "SamplingArgs",
    "StartCallback",
    "State",
    "SystemMessage",
    "TextContentPart",
    "TextMessage",
    "ThinkingBlock",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "TrajectoryStep",
    "TrajectoryStepTokens",
    "Usage",
    "UserMessage",
    "VersionInfo",
}
__all__ += sorted(_TYPE_EXPORTS.difference(__all__))

_LAZY_IMPORTS = {
    "Parser": "verifiers.parsers.parser:Parser",
    "ThinkParser": "verifiers.parsers.think_parser:ThinkParser",
    "MaybeThinkParser": "verifiers.parsers.maybe_think_parser:MaybeThinkParser",
    "XMLParser": "verifiers.parsers.xml_parser:XMLParser",
    "Rubric": "verifiers.rubrics.rubric:Rubric",
    "JudgeRubric": "verifiers.rubrics.judge_rubric:JudgeRubric",
    "RubricGroup": "verifiers.rubrics.rubric_group:RubricGroup",
    "Environment": "verifiers.envs.environment:Environment",
    "MultiTurnEnv": "verifiers.envs.multiturn_env:MultiTurnEnv",
    "SingleTurnEnv": "verifiers.envs.singleturn_env:SingleTurnEnv",
    "StatefulToolEnv": "verifiers.envs.stateful_tool_env:StatefulToolEnv",
    "ToolEnv": "verifiers.envs.tool_env:ToolEnv",
    "EnvGroup": "verifiers.envs.env_group:EnvGroup",
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
    "extract_boxed_answer": "verifiers.utils.data_utils:extract_boxed_answer",
    "extract_hash_answer": "verifiers.utils.data_utils:extract_hash_answer",
    "load_example_dataset": "verifiers.utils.data_utils:load_example_dataset",
    "setup_logging": "verifiers.utils.logging_utils:setup_logging",
    "log_level": "verifiers.utils.logging_utils:log_level",
    "quiet_verifiers": "verifiers.utils.logging_utils:quiet_verifiers",
    "print_prompt_completions_sample": (
        "verifiers.utils.logging_utils:print_prompt_completions_sample"
    ),
    "load_environment": "verifiers.utils.env_utils:load_environment",
    "cleanup": "verifiers.decorators:cleanup",
    "stop": "verifiers.decorators:stop",
    "teardown": "verifiers.decorators:teardown",
    "ensure_keys": "verifiers.utils.config_utils:ensure_keys",
    "MissingKeyError": "verifiers.utils.config_utils:MissingKeyError",
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
_LAZY_IMPORTS.update({name: f"verifiers.types:{name}" for name in _TYPE_EXPORTS})


def __getattr__(name: str):
    try:
        module, attr = _LAZY_IMPORTS[name].split(":")
    except KeyError as exc:
        raise AttributeError(f"module 'verifiers' has no attribute '{name}'") from exc

    try:
        return getattr(importlib.import_module(module), attr)
    except ModuleNotFoundError as exc:
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
            ) from exc
        raise AttributeError(
            f"To use verifiers.{name}, install as `verifiers[all]`."
        ) from exc


if TYPE_CHECKING:
    from typing import Any

    from .clients.anthropic_messages_client import AnthropicMessagesClient
    from .clients.client import Client
    from .clients.openai_chat_completions_client import OpenAIChatCompletionsClient
    from .clients.openai_chat_completions_token_client import (
        OpenAIChatCompletionsTokenClient,
    )
    from .clients.openai_completions_client import OpenAICompletionsClient
    from .decorators import cleanup, stop, teardown
    from .envs.env_group import EnvGroup
    from .envs.environment import Environment
    from .envs.experimental.cli_agent_env import CliAgentEnv
    from .envs.experimental.gym_env import GymEnv
    from .envs.experimental.harbor_env import HarborEnv
    from .envs.experimental.mcp_env import MCPEnv
    from .envs.integrations.browser_env import BrowserEnv
    from .envs.integrations.openenv_env import OpenEnvEnv
    from .envs.integrations.reasoninggym_env import ReasoningGymEnv
    from .envs.integrations.textarena_env import TextArenaEnv
    from .envs.multiturn_env import MultiTurnEnv
    from .envs.python_env import PythonEnv
    from .envs.sandbox_env import SandboxEnv
    from .envs.singleturn_env import SingleTurnEnv
    from .envs.stateful_tool_env import StatefulToolEnv
    from .envs.tool_env import ToolEnv
    from .parsers.maybe_think_parser import MaybeThinkParser
    from .parsers.parser import Parser
    from .parsers.think_parser import ThinkParser
    from .parsers.xml_parser import XMLParser
    from .rubrics.judge_rubric import JudgeRubric
    from .rubrics.math_rubric import MathRubric
    from .rubrics.rubric import Rubric
    from .rubrics.rubric_group import RubricGroup
    from .types import DatasetBuilder
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

    RLConfig: Any
    RLTrainer: Any
    GRPOTrainer: Any
    GRPOConfig: Any
    grpo_defaults: Any
    lora_defaults: Any
    get_model: Any
    get_model_and_tokenizer: Any
