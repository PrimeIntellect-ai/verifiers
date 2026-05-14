import re

import verifiers as vf
from verifiers.types import Messages, UserMessage


class TextArenaPromptRenderer:
    env_id_pattern = re.compile(r"^[A-Za-z0-9_-]+-v\d+$")

    def __call__(
        self,
        observation: object,
        *,
        context: str = "reset",
    ) -> Messages:
        if not isinstance(observation, dict):
            raise RuntimeError(
                f"openenv-textarena prompt renderer expected dict observation, got {type(observation).__name__}."
            )

        message_text = self.message_text(observation)
        prompt_text = self.prompt_text(observation)

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

    def message_text(self, observation: vf.ConfigData) -> str | None:
        raw_messages = observation.get("messages")
        if not isinstance(raw_messages, list):
            return None
        for item in reversed(raw_messages):
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return None

    def prompt_text(self, observation: vf.ConfigData) -> str | None:
        prompt = observation.get("prompt")
        if not isinstance(prompt, str):
            return None
        value = prompt.strip()
        if not value:
            return None
        # TextArena sometimes falls back to env id like "Wordle-v0", which is
        # not a useful model prompt for subsequent turns.
        if self.env_id_pattern.fullmatch(value):
            return None
        return value


def load_environment(
    num_train_examples: int = 100,
    num_eval_examples: int = 50,
    seed: int = 0,
):
    return vf.OpenEnvEnv(
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        prompt_renderer=TextArenaPromptRenderer(),
    )
