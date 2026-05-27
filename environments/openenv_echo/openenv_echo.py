from collections.abc import Mapping
from typing import cast

import verifiers as vf
from tasksets.openenv import OpenEnvTaskset, OpenEnvTasksetConfig
from verifiers.types import Messages, UserMessage
from verifiers.utils.message_utils import MessageInput, normalize_messages


class OpenEnvEchoTasksetConfig(OpenEnvTasksetConfig):
    prompt_renderer: str = "openenv_echo:render_openenv_prompt"


def render_openenv_prompt(
    observation: object,
    *,
    action_schema: vf.ConfigData | None = None,
    context: str = "reset",
) -> Messages:
    if not isinstance(observation, Mapping):
        raise RuntimeError(
            f"openenv-echo prompt renderer expected dict observation, got {type(observation).__name__}."
        )
    observation_data = cast(vf.ConfigData, observation)

    messages = observation_data.get("messages")
    if isinstance(messages, list) and messages:
        try:
            return normalize_messages(
                cast(MessageInput, messages),
                field_name="openenv-echo observation messages",
            )
        except TypeError as e:
            raise RuntimeError(str(e)) from e

    prompt = observation_data.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return [UserMessage(content=prompt)]

    if context == "reset" and isinstance(action_schema, dict):
        return [
            UserMessage(
                content=(
                    "You are connected to an OpenEnv MCP environment. "
                    "Call at least one tool before your final response. "
                    "Action contract: call_tool(tool_name: str, arguments: object)."
                )
            )
        ]

    raise RuntimeError("openenv-echo observation did not include a renderable prompt.")


def load_taskset(config: OpenEnvEchoTasksetConfig) -> OpenEnvTaskset:
    return OpenEnvTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=vf.load_taskset(config=config.taskset))
