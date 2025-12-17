import logging
from typing import Any, Callable

from verifiers.types import ChatMessage, Messages


class Parser:
    """
    Parser class for parsing LLM rollouts.

    Default behavior:
    - `parse` returns text as-is
    - `parse_answer` returns the last message's content (or text if string)
    """

    def __init__(self, extract_fn: Callable[[str], str] = lambda x: x):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.extract_fn = extract_fn

    def parse(self, text: str) -> Any:
        return self.extract_fn(text)

    def get_assistant_messages(
        self, completion: list[ChatMessage]
    ) -> list[ChatMessage]:
        """Helper function to extract assistant messages from a completion."""
        return [msg for msg in completion if msg["role"] == "assistant"]

    def get_system_messages(self, completion: list[ChatMessage]) -> list[ChatMessage]:
        """Helper function to extract system messages from a completion."""
        return [msg for msg in completion if msg["role"] == "system"]

    def get_user_messages(self, completion: list[ChatMessage]) -> list[ChatMessage]:
        """Helper function to extract user messages from a completion."""
        return [msg for msg in completion if msg["role"] == "user"]

    def get_tool_messages(self, completion: list[ChatMessage]) -> list[ChatMessage]:
        """Helper function to extract tool messages from a completion."""
        return [msg for msg in completion if msg["role"] == "tool"]

    def parse_answer(self, completion: Messages) -> str | None:
        if isinstance(completion, str):
            return self.parse(completion)
        else:
            ans = str(self.get_assistant_messages(completion)[-1].get("content", ""))
            return self.parse(ans)

    def get_format_reward_func(self) -> Callable:
        """
        Reward function that checks if the final answer is formatted correctly.
        """

        def format_reward_func(completion: list[dict[str, str]], **kwargs) -> float:
            return 1.0

        return format_reward_func
