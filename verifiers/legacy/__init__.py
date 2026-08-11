# ruff: noqa

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
from .errors import *  # noqa # isort: skip
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
    "DatasetBuilder",
    "State",
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
    "EndpointConfig",
    "Endpoints",
    "TaskSplit",
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
    "print_prompt_completions_sample",
    "cleanup",
    "metric",
    "reward",
    "advantage",
    "setup",
    "stop",
    "teardown",
    "update",
    "ensure_keys",
    "MissingKeyError",
]

_LAZY_IMPORTS = {
    "Client": "verifiers.legacy.clients.client:Client",
    "AnthropicMessagesClient": (
        "verifiers.legacy.clients.anthropic_messages_client:AnthropicMessagesClient"
    ),
    "OpenAIChatCompletionsClient": (
        "verifiers.legacy.clients.openai_chat_completions_client:OpenAIChatCompletionsClient"
    ),
    "RendererClient": ("verifiers.legacy.clients.renderer_client:RendererClient"),
    "OpenAICompletionsClient": (
        "verifiers.legacy.clients.openai_completions_client:OpenAICompletionsClient"
    ),
    "OpenAIResponsesClient": (
        "verifiers.legacy.clients.openai_responses_client:OpenAIResponsesClient"
    ),
    "Environment": "verifiers.legacy.envs.environment:Environment",
    "MultiTurnEnv": "verifiers.legacy.envs.multiturn_env:MultiTurnEnv",
    "SingleTurnEnv": "verifiers.legacy.envs.singleturn_env:SingleTurnEnv",
    "StatefulToolEnv": "verifiers.legacy.envs.stateful_tool_env:StatefulToolEnv",
    "ToolEnv": "verifiers.legacy.envs.tool_env:ToolEnv",
    "EnvGroup": "verifiers.legacy.envs.env_group:EnvGroup",
    "JudgeRubric": "verifiers.legacy.rubrics.judge_rubric:JudgeRubric",
    "load_environment": "verifiers.legacy.utils.env_utils:load_environment",
    "MathRubric": "verifiers.legacy.rubrics.math_rubric:MathRubric",
    "SandboxEnv": "verifiers.legacy.envs.sandbox_env:SandboxEnv",
    "PythonEnv": "verifiers.legacy.envs.python_env:PythonEnv",
    "GymEnv": "verifiers.legacy.envs.experimental.gym_env:GymEnv",
    "CliAgentEnv": "verifiers.legacy.envs.experimental.cli_agent_env:CliAgentEnv",
    "HarborEnv": "verifiers.legacy.envs.experimental.harbor_env:HarborEnv",
    "MCPEnv": "verifiers.legacy.envs.experimental.mcp_env:MCPEnv",
    "ReasoningGymEnv": "verifiers.legacy.envs.integrations.reasoninggym_env:ReasoningGymEnv",
    "TextArenaEnv": "verifiers.legacy.envs.integrations.textarena_env:TextArenaEnv",
    "BrowserEnv": "verifiers.legacy.envs.integrations.browser_env:BrowserEnv",
    "OpenEnvEnv": "verifiers.legacy.envs.integrations.openenv_env:OpenEnvEnv",
}


def __getattr__(name: str):
    try:
        module, attr = _LAZY_IMPORTS[name].split(":")
        return getattr(importlib.import_module(module), attr)
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    except ModuleNotFoundError as e:
        if name == "RendererClient":
            raise
        raise AttributeError(
            f"To use verifiers.{name}, install as `verifiers[all]`. "
        ) from e


if TYPE_CHECKING:
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
    from .utils.env_utils import load_environment  # noqa: F401
