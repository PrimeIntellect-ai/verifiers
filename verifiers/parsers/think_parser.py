from typing import Callable

from verifiers.parsers.parser import Parser
from verifiers.types import ChatMessage


class ThinkParser(Parser):
    def __init__(self, extract_fn: Callable[[str], str] = lambda x: x):
        super().__init__(extract_fn=extract_fn)
        self.extract_fn = extract_fn
        self.logger.warning(
            "ThinkParser is intended for use with models which always include <think>...</think> tags but do NOT parse them automatically. "
            "This will cause parsing failures if the model does not include <think>...</think> tags, or if the chat template automatically removes <think>...</think> tags."
            "In particular, you should NOT use this parser with Qwen3 or DeepSeek-R1 models."
        )

    def parse(self, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        else:  # do not allow any further extraction/ parsing if no </think> is found
            text = ""
        return self.extract_fn(text.strip())

    def get_format_reward_func(self) -> Callable:
        """
        Return a reward function that checks if each message follows the format:
        <think>
        ...
        </think>
        ...
        """

        def follows_format(text: str) -> float:
            if (
                text.strip().startswith("<think>")
                and text.count("<think>") == 1
                and text.count("</think>") == 1
                and len(text.split("</think>")[-1]) > 0
            ):
                return 1.0
            return 0.0

        def format_reward_func(completion: list[ChatMessage], **kwargs) -> float:
            messages = self.get_assistant_messages(completion)
            return sum(follows_format(m["content"]) for m in messages) / len(messages)  # type: ignore

        return format_reward_func
