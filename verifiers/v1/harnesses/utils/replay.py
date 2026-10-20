"""Cosmetic rewrites agent CLIs apply when they replay an earlier assistant turn, for
`Harness.replay_view` overrides. Each function is idempotent and only normalizes
whitespace, so the replayed and the sampled form of a turn map to the same view."""

from verifiers.v1.types import AssistantMessage, Message


def drop_blank_text(message: Message) -> Message:
    """Clients that filter text blocks with `text.trim()` drop whitespace-only
    assistant text, such as the "\\n\\n" some models emit before a tool call."""
    if (
        isinstance(message, AssistantMessage)
        and isinstance(message.content, str)
        and not message.content.strip()
    ):
        return message.model_copy(update={"content": None})
    return message


def strip_text(message: Message) -> Message:
    """Clients that store assistant text stripped, and pad a missing reasoning field
    (or Anthropic thinking block) with a single space on replay."""
    if not isinstance(message, AssistantMessage):
        return message
    update: dict = {}
    if isinstance(message.content, str):
        update["content"] = message.content.strip() or None
    if message.reasoning_content is not None and not message.reasoning_content.strip():
        update["reasoning_content"] = None
    if message.provider_state and any(
        _blank_thinking(item) for item in message.provider_state
    ):
        update["provider_state"] = [
            item for item in message.provider_state if not _blank_thinking(item)
        ] or None
    return message.model_copy(update=update) if update else message


def _blank_thinking(item: dict) -> bool:
    return item.get("type") == "thinking" and not str(item.get("thinking", "")).strip()
