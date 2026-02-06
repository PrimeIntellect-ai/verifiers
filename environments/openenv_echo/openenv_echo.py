from pathlib import Path
from typing import Any, cast

import verifiers as vf
from verifiers.types import ChatMessages


def render_echo_prompt(
    observation: Any,
    *,
    action_schema: dict[str, Any] | None = None,
    context: str = "reset",
    **kwargs: Any,
) -> ChatMessages:
    del kwargs
    if not isinstance(observation, dict):
        raise RuntimeError(
            f"openenv-echo prompt renderer expected dict observation, got {type(observation).__name__}."
        )

    messages = observation.get("messages")
    if isinstance(messages, list) and messages:
        return cast(ChatMessages, messages)

    prompt = observation.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return cast(ChatMessages, [{"role": "user", "content": prompt}])

    if context == "reset" and isinstance(action_schema, dict):
        return cast(
            ChatMessages,
            [
                {
                    "role": "user",
                    "content": (
                        "You are connected to an OpenEnv MCP environment. "
                        "Call at least one tool before your final response. "
                        "Action contract: call_tool(tool_name: str, arguments: object)."
                    ),
                }
            ],
        )

    raise RuntimeError("openenv-echo observation did not include a renderable prompt.")


def load_environment(
    num_train_examples: int = 100,
    num_eval_examples: int = 50,
    seed: int = 0,
):
    return vf.OpenEnvEnv(
        openenv_project=Path(__file__).parent / "proj",
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        prompt_renderer=render_echo_prompt,
    )
