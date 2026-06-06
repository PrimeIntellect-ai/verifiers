import re
from collections.abc import Mapping
from typing import cast

import verifiers.v1 as vf
from tasksets.openenv import OpenEnvTaskset, OpenEnvTasksetConfig
from verifiers.types import Messages, UserMessage


ENV_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+-v\d+$")


class OpenEnvTextArenaTasksetConfig(OpenEnvTasksetConfig):
    prompt_renderer: str = "openenv_textarena_v1.taskset:render_textarena_prompt"


def render_textarena_prompt(
    observation: object,
    *,
    context: str = "reset",
    action_schema: vf.JsonData | None = None,
    contract: str = "gym",
    seed: int = 0,
) -> Messages:
    del action_schema, contract, seed
    if not isinstance(observation, Mapping):
        raise RuntimeError(
            f"openenv-textarena prompt renderer expected dict observation, got {type(observation).__name__}."
        )
    observation_data = cast(vf.JsonData, observation)

    message_text = textarena_message_text(observation_data)
    prompt_text = textarena_prompt_text(observation_data)

    if context == "step":
        if message_text is not None:
            return [UserMessage(content=message_text)]
        if prompt_text is not None:
            return [UserMessage(content=prompt_text)]
    else:
        if prompt_text is not None:
            return [UserMessage(content=prompt_text)]
        if message_text is not None:
            return [UserMessage(content=message_text)]

    raise RuntimeError(
        "openenv-textarena observation did not include renderable prompt text."
    )


def textarena_message_text(observation: vf.JsonData) -> str | None:
    raw_messages = observation.get("messages")
    if not isinstance(raw_messages, list):
        return None
    for item in reversed(raw_messages):
        if isinstance(item, Mapping):
            message = cast(vf.JsonData, item)
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def textarena_prompt_text(observation: vf.JsonData) -> str | None:
    prompt = observation.get("prompt")
    if not isinstance(prompt, str):
        return None
    value = prompt.strip()
    if not value:
        return None
    # TextArena sometimes falls back to env id like "Wordle-v0", which is
    # not a useful model prompt for subsequent turns.
    if ENV_ID_PATTERN.fullmatch(value):
        return None
    return value


def load_taskset(config: OpenEnvTextArenaTasksetConfig) -> OpenEnvTaskset:
    return OpenEnvTaskset(config=config)
