from collections.abc import Mapping
from typing import cast

import verifiers.v1 as vf
from tasksets.openenv import OpenEnvTaskset, OpenEnvTasksetConfig


class OpenEnvEchoTasksetConfig(OpenEnvTasksetConfig):
    prompt_renderer: str = "openenv_echo_v1.taskset:render_openenv_prompt"


def render_openenv_prompt(
    observation: object,
    *,
    action_schema: vf.JsonData | None = None,
    context: str = "reset",
    contract: str = "mcp",
    seed: int = 0,
) -> vf.Messages:
    del contract, seed
    if not isinstance(observation, Mapping):
        raise RuntimeError(
            f"openenv-echo prompt renderer expected dict observation, got {type(observation).__name__}."
        )
    observation_data = cast(vf.JsonData, observation)

    messages = observation_data.get("messages")
    if isinstance(messages, list) and messages:
        parsed: vf.Messages = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise RuntimeError("openenv-echo observation messages must be objects.")
            payload = dict(message)
            role = payload.get("role")
            if role == "user":
                parsed.append(vf.UserMessage.model_validate(payload))
            elif role == "assistant":
                parsed.append(vf.AssistantMessage.model_validate(payload))
            elif role == "system":
                parsed.append(vf.SystemMessage.model_validate(payload))
            elif role == "tool":
                parsed.append(vf.ToolMessage.model_validate(payload))
            else:
                raise RuntimeError(f"Unsupported openenv-echo message role: {role!r}.")
        return parsed

    prompt = observation_data.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return [vf.UserMessage(content=prompt)]

    if context == "reset" and isinstance(action_schema, dict):
        return [
            vf.UserMessage(
                content=(
                    "You are connected to an OpenEnv MCP environment. "
                    "Call user_call_tool before your final response. "
                    "Available tool names are echo_message and echo_with_length. "
                    "Use user_call_tool(name: str, input: object)."
                )
            )
        ]

    raise RuntimeError("openenv-echo observation did not include a renderable prompt.")


def load_taskset(config: OpenEnvEchoTasksetConfig) -> OpenEnvTaskset:
    return OpenEnvTaskset(config=config)
