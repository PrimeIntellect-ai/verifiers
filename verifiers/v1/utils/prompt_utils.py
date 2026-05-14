from collections.abc import Mapping, Sequence
from typing import Literal, cast

from verifiers.types import MessageContent, Messages, SystemMessage
from verifiers.utils.message_utils import MessageLike, get_messages, normalize_messages
from ..types import ConfigData, ConfigMap, PromptInput


SystemPromptMerge = Literal["reject", "concat", "task", "taskset", "harness"]


def normalize_prompt(
    value: PromptInput | None, field_name: str = "prompt"
) -> list[ConfigData]:
    messages = normalize_messages(cast(Messages, value or []), field_name=field_name)
    for message in messages:
        if getattr(message, "role", None) == "system":
            raise ValueError(
                f"{field_name} must not contain system messages. "
                "Use system_prompt instead."
            )
    return dump_messages(messages)


def normalize_system_prompt(
    value: PromptInput | None, field_name: str = "system_prompt"
) -> list[ConfigData]:
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
    task: ConfigMap,
    taskset_system_prompt: list[ConfigData],
    harness_system_prompt: list[ConfigData],
    merge: str,
) -> list[ConfigData]:
    task_system_prompt = normalize_system_prompt(
        cast(PromptInput | None, task.get("system_prompt")),
        field_name="task.system_prompt",
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


def dump_messages(messages: Messages) -> list[ConfigData]:
    return [message.model_dump(exclude_none=True) for message in messages]


def task_text(
    task: ConfigMap,
    state: ConfigMap,
    *,
    keys: tuple[str, ...] = ("instruction",),
) -> str:
    _ = state
    for key in keys:
        value = task.get(key)
        if isinstance(value, str) and value:
            return value
    return messages_text(task.get("prompt", []))


def task_question_text(task: ConfigMap) -> str:
    question = task.get("question")
    if question is not None:
        return str(question)
    prompt = task.get("prompt")
    if isinstance(prompt, list) and prompt:
        messages = get_messages(cast(Sequence[MessageLike], prompt))
        return content_text(messages[-1].content) if messages else ""
    return ""


def state_system_prompt_text(task: ConfigMap, state: ConfigMap) -> str:
    _ = task
    return messages_text(state.get("system_prompt", []))


def state_result_text(state: ConfigMap, *, result_key: str = "agent_result") -> str:
    result = state.get(result_key)
    if result is not None:
        return str(result)
    completion = state.get("completion")
    if not isinstance(completion, list):
        return ""
    completion_messages = cast(Sequence[MessageLike], completion)
    messages = get_messages(completion_messages, role="assistant") or get_messages(
        completion_messages
    )
    return content_text(messages[-1].content) if messages else ""


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
            item = cast(ConfigMap, message)
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
                item = cast(ConfigMap, part)
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    return str(content)
