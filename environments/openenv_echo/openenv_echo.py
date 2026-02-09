from typing import Any, cast

import verifiers as vf
from verifiers.types import Message, Messages, UserMessage
from verifiers.utils.message_utils import from_raw_message


def render_echo_prompt(
    observation: Any,
    *,
    action_schema: dict[str, Any] | None = None,
    context: str = "reset",
) -> Messages:
    if not isinstance(observation, dict):
        raise RuntimeError(
            f"openenv-echo prompt renderer expected dict observation, got {type(observation).__name__}."
        )

    messages = observation.get("messages")
    if isinstance(messages, list) and messages:
        normalized: Messages = []
        for message in messages:
            if isinstance(message, dict):
                normalized.append(from_raw_message(dict(message)))
                continue
            if hasattr(message, "role") and hasattr(message, "content"):
                normalized.append(cast(Message, message))
                continue
            raise RuntimeError(
                "openenv-echo observation contains unsupported message item type: "
                f"{type(message).__name__}."
            )
        return normalized

    prompt = observation.get("prompt")
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


def load_environment(
    num_train_examples: int = 100,
    num_eval_examples: int = 50,
    seed: int = 0,
):
    return vf.OpenEnvEnv(
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        prompt_renderer=render_echo_prompt,
    )
