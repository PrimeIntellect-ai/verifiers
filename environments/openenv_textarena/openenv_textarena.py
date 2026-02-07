import re
from typing import Any, cast

import verifiers as vf
from verifiers.types import ChatMessages


_TEXTARENA_ENV_ID_RE = re.compile(r"^[A-Za-z0-9_-]+-v\d+$")


def _message_text_from_observation(observation: dict[str, Any]) -> str | None:
    raw_messages = observation.get("messages")
    if not isinstance(raw_messages, list):
        return None
    for item in reversed(raw_messages):
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _prompt_text_from_observation(observation: dict[str, Any]) -> str | None:
    prompt = observation.get("prompt")
    if not isinstance(prompt, str):
        return None
    value = prompt.strip()
    if not value:
        return None
    # TextArena sometimes falls back to env id like "Wordle-v0", which is not
    # a useful model prompt for subsequent turns.
    if _TEXTARENA_ENV_ID_RE.fullmatch(value):
        return None
    return value


def render_textarena_prompt(
    observation: Any,
    *,
    context: str = "reset",
) -> ChatMessages:
    if not isinstance(observation, dict):
        raise RuntimeError(
            f"openenv-textarena prompt renderer expected dict observation, got {type(observation).__name__}."
        )

    message_text = _message_text_from_observation(observation)
    prompt_text = _prompt_text_from_observation(observation)

    if context == "step":
        if message_text is not None:
            return cast(ChatMessages, [{"role": "user", "content": message_text}])
        if prompt_text is not None:
            return cast(ChatMessages, [{"role": "user", "content": prompt_text}])
    else:
        if prompt_text is not None:
            return cast(ChatMessages, [{"role": "user", "content": prompt_text}])
        if message_text is not None:
            return cast(ChatMessages, [{"role": "user", "content": message_text}])

    raise RuntimeError(
        "openenv-textarena observation did not include renderable prompt text."
    )


def load_environment(
    num_train_examples: int = 100,
    num_eval_examples: int = 50,
    seed: int = 0,
):
    return vf.OpenEnvEnv(
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        prompt_renderer=render_textarena_prompt,
    )
