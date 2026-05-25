from collections.abc import Callable, Mapping
import importlib.util
from os import PathLike
from pathlib import Path
import re
from typing import Literal, cast

from verifiers.types import MessageContent, Messages, SystemMessage
from verifiers.utils.message_utils import normalize_messages
from ..types import ConfigData, ConfigMap, PromptInput
from .config_utils import current_config_ref_module, import_config_ref


SystemPromptMerge = Literal["reject", "concat", "task", "taskset", "harness"]
SystemPromptLoader = Callable[[], PromptInput | PathLike[str] | None]


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
    value = resolve_system_prompt_input(value, field_name=field_name)
    if value is None:
        return []
    if isinstance(value, str):
        return [SystemMessage(content=value).model_dump(exclude_none=True)]
    messages = normalize_messages(cast(Messages, value), field_name=field_name)
    for message in messages:
        if getattr(message, "role", None) != "system":
            raise ValueError(f"{field_name} accepts only system messages.")
    return dump_messages(messages)


def resolve_system_prompt_input(
    value: PromptInput | PathLike[str] | None,
    *,
    field_name: str,
) -> PromptInput | None:
    if value is None:
        return None
    if not isinstance(value, str):
        if isinstance(value, PathLike):
            path = Path(cast(PathLike[str], value))
            return read_system_prompt_path(path, field_name=field_name)
        return value
    if is_system_prompt_path(value):
        return read_system_prompt_path(resolve_system_prompt_path(value), field_name)
    if is_system_prompt_ref(value):
        resolved = import_config_ref(value)
        if callable(resolved):
            resolved = cast(SystemPromptLoader, resolved)()
        if isinstance(resolved, PathLike):
            path = Path(cast(PathLike[str], resolved))
            return read_system_prompt_path(path, field_name=field_name)
        return cast(PromptInput | None, resolved)
    return value


def is_system_prompt_ref(value: str) -> bool:
    if any(character.isspace() for character in value):
        return False
    if re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*",
        value,
    ):
        _, _, attr_path = value.partition(":")
        return is_system_prompt_ref_name(attr_path.rsplit(".", 1)[-1])
    if current_config_ref_module() is None:
        return False
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", value):
        return False
    return is_system_prompt_ref_name(value.rsplit(".", 1)[-1])


def is_system_prompt_ref_name(value: str) -> bool:
    return (
        value.startswith("load_")
        or value.endswith("_prompt")
        or value.endswith("_PROMPT")
        or value.endswith("PROMPT")
    )


def is_system_prompt_path(value: str) -> bool:
    if any(character.isspace() for character in value):
        return False
    path = Path(value).expanduser()
    return (
        path.is_absolute()
        or value.startswith(("./", "../", "~/"))
        or "/" in value
        or path.suffix in {".txt", ".md", ".rst"}
    )


def resolve_system_prompt_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    module_name = current_config_ref_module()
    if module_name is None:
        return path
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return path
    return Path(spec.origin).parent / path


def read_system_prompt_path(path: Path, field_name: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{field_name} path {str(path)!r} could not be read.") from exc


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


def state_system_prompt_text(task: ConfigMap, state: ConfigMap) -> str:
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
