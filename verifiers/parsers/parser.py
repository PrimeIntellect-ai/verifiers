import logging
from typing import Any, Callable

from verifiers.types import Messages


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

    def _message_field(self, message: Any, field: str, default: Any = None) -> Any:
        if isinstance(message, dict):
            return message.get(field, default)
        return getattr(message, field, default)

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        text = part.get("text")
                        if isinstance(text, str):
                            chunks.append(text)
                    continue
                text_attr = getattr(part, "text", None)
                if isinstance(text_attr, str):
                    chunks.append(text_attr)
            return " ".join(chunks).strip()
        return ""

    def get_assistant_messages(self, completion: Messages) -> Messages:
        """Helper function to extract assistant messages from a completion."""
        return [
            msg for msg in completion if self._message_field(msg, "role") == "assistant"
        ]

    def get_system_messages(self, completion: Messages) -> Messages:
        """Helper function to extract system messages from a completion."""
        return [
            msg for msg in completion if self._message_field(msg, "role") == "system"
        ]

    def get_user_messages(self, completion: Messages) -> Messages:
        """Helper function to extract user messages from a completion."""
        return [msg for msg in completion if self._message_field(msg, "role") == "user"]

    def get_tool_messages(self, completion: Messages) -> Messages:
        """Helper function to extract tool messages from a completion."""
        return [msg for msg in completion if self._message_field(msg, "role") == "tool"]

    def parse_answer(self, completion: Messages) -> str | None:
        if isinstance(completion, str):
            return self.parse(completion)
        else:
            assistant_messages = self.get_assistant_messages(completion)
            if not assistant_messages:
                return None
            ans = self._message_field(assistant_messages[-1], "content", "") or ""
            return self.parse(self._content_to_text(ans))

    def get_format_reward_func(self) -> Callable:
        """
        Reward function that checks if the final answer is formatted correctly.
        """

        def format_reward_func(completion: list[dict[str, str]], **kwargs) -> float:
            return 1.0

        return format_reward_func
