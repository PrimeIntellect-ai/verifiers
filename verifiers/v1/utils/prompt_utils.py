from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from verifiers.types import MessageContent, Messages, SystemMessage
from verifiers.utils.message_utils import normalize_messages


SystemPromptMerge = Literal["reject", "concat", "task", "taskset", "harness"]


def normalize_prompt(
    value: object, field_name: str = "prompt"
) -> list[dict[str, object]]:
    messages = normalize_messages(cast(Messages, value or []), field_name=field_name)
    for message in messages:
        if getattr(message, "role", None) == "system":
            raise ValueError(
                f"{field_name} must not contain system messages. "
                "Use system_prompt instead."
            )
    return dump_messages(messages)


def normalize_system_prompt(
    value: object, field_name: str = "system_prompt"
) -> list[dict[str, object]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [SystemMessage(content=value).model_dump(exclude_none=True)]
    messages = normalize_messages(cast(Messages, value), field_name=field_name)
    for message in messages:
        if getattr(message, "role", None) != "system":
            raise ValueError(f"{field_name} accepts only system messages.")
    return dump_messages(messages)


def resolve_system_prompt(
    *,
    task: Mapping[str, object],
    taskset_system_prompt: list[dict[str, object]],
    harness_system_prompt: list[dict[str, object]],
    merge: str,
) -> list[dict[str, object]]:
    task_system_prompt = normalize_system_prompt(
        task.get("system_prompt"), field_name="task.system_prompt"
    )
    sources = [
        ("harness", harness_system_prompt),
        ("taskset", taskset_system_prompt),
        ("task", task_system_prompt),
    ]
    present = [(name, messages) for name, messages in sources if messages]

    if merge == "reject":
        if len(present) > 1:
            names = ", ".join(name for name, _ in present)
            raise ValueError(
                f"Multiple system_prompt sources cannot be resolved: {names}. "
                "Set system_prompt_merge='concat' or choose one source."
            )
        return [dict(message) for _, messages in present for message in messages]
    if merge == "concat":
        return [dict(message) for _, messages in present for message in messages]
    if merge in {"task", "taskset", "harness"}:
        for name, messages in present:
            if name == merge:
                return [dict(message) for message in messages]
        return []
    raise ValueError(
        "system_prompt_merge must be one of reject, concat, task, taskset, harness."
    )


def dump_messages(messages: Messages) -> list[dict[str, object]]:
    return [message.model_dump(exclude_none=True) for message in messages]


def task_text(
    task: Mapping[str, object],
    state: Mapping[str, object],
    *,
    keys: tuple[str, ...] = ("instruction",),
) -> str:
    _ = state
    for key in keys:
        value = task.get(key)
        if isinstance(value, str) and value:
            return value
    return messages_text(task.get("prompt", []))


def state_system_prompt_text(
    task: Mapping[str, object], state: Mapping[str, object]
) -> str:
    _ = task
    return messages_text(state.get("system_prompt", []))


def messages_text(messages: object) -> str:
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, list):
        return str(messages or "")
    parts: list[str] = []
    for message in messages:
        content = getattr(message, "content", None)
        if content is not None:
            parts.append(content_text(content))
        elif isinstance(message, Mapping):
            item = cast(Mapping[str, object], message)
            parts.append(content_text(item.get("content")))
        else:
            parts.append(str(message))
    return "\n\n".join(part for part in parts if part)


def content_text(content: MessageContent | object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                item = cast(Mapping[str, object], part)
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    return str(content)
