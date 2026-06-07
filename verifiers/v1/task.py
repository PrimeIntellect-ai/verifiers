from __future__ import annotations

import uuid
from collections.abc import Mapping
from copy import deepcopy
from typing import TypeVar, cast

from pydantic import (
    BaseModel,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
    model_validator,
)
from typing_extensions import Self

from verifiers.types import Messages, UserMessage

from .types import JsonData
from .utils.prompt_utils import SystemPrompt, dump_messages, normalize_system_prompt
from .utils.task_freeze_utils import assert_serializable

MESSAGES_ADAPTER = TypeAdapter(Messages)


class TaskVisibility(BaseModel, extra="forbid", frozen=True):
    show: list[str] | None = None
    hide: list[str] | None = None

    @model_validator(mode="after")
    def validate_visibility(self) -> Self:
        if self.show is not None and self.hide is not None:
            raise ValueError("Task visibility accepts show or hide, not both.")
        return self


class Task(BaseModel, extra="forbid", frozen=True):
    """Immutable serializable task specification. Subclass for task-specific data."""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    row_id: int = 0
    prompt: Messages = Field(default_factory=list)
    system_prompt: SystemPrompt = None
    toolsets: TaskVisibility | None = None
    tools: TaskVisibility | None = None
    user: bool | None = None
    name: str | None = None
    description: str | None = None
    image: str | None = None
    max_turns: int | None = None

    def __init__(
        self, task: Mapping[str, object] | None = None, **data: object
    ) -> None:
        if task is not None and data:
            raise TypeError(
                "Task accepts either a mapping or keyword fields, not both."
            )
        super().__init__(**deepcopy(dict(task or data)))

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, value: object) -> object:
        if isinstance(value, Task):
            return deepcopy(value.model_dump(mode="python"))
        if isinstance(value, Mapping):
            raw = deepcopy(dict(value))
            if "id" in raw and "task_id" not in raw:
                raw["task_id"] = raw.pop("id")
            if "example_id" in raw and "row_id" not in raw:
                raw["row_id"] = raw.pop("example_id")
            raw.pop("example_id", None)
            if "prompt" in raw:
                raw["prompt"] = normalized_task_prompt(raw["prompt"])
            if "system_prompt" in raw:
                raw["system_prompt"] = normalized_task_system_prompt(
                    raw["system_prompt"]
                )
            return raw
        return value

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        if isinstance(self.row_id, bool):
            raise TypeError("task.row_id must be an integer.")
        if self.max_turns is not None and (
            isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int)
        ):
            raise TypeError("task.max_turns must be an integer.")
        object.__setattr__(self, "prompt", normalized_task_prompt(self.prompt))
        object.__setattr__(
            self,
            "system_prompt",
            normalized_task_system_prompt(self.system_prompt),
        )
        assert_serializable(self.model_dump(mode="json", exclude_none=True))
        return self

    @model_serializer(mode="wrap")
    def serialize_task(
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> dict[str, object]:
        data = handler(self)
        if not isinstance(data, dict):
            raise TypeError("Task serializer expected a JSON object.")
        if info.mode != "json":
            return cast(dict[str, object], data)
        if "prompt" in data:
            data["prompt"] = dump_messages(self.prompt)
        if "system_prompt" in data:
            if self.system_prompt:
                data["system_prompt"] = list(
                    normalized_task_system_prompt(self.system_prompt)
                )
            else:
                data.pop("system_prompt", None)
        return cast(dict[str, object], data)


TaskT = TypeVar("TaskT", bound=Task)


def normalized_task_prompt(value: object) -> Messages:
    messages = (
        [UserMessage(content=value)]
        if isinstance(value, str)
        else MESSAGES_ADAPTER.validate_python(value or [])
    )
    for message in messages:
        if getattr(message, "role", None) == "system":
            raise ValueError("task.prompt must not contain system messages.")
    return messages


def normalized_task_system_prompt(value: object) -> list[JsonData]:
    return normalize_system_prompt(
        task_system_prompt_input(value),
        field_name="task.system_prompt",
    )


def task_system_prompt_input(value: object) -> SystemPrompt:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return value
    from .utils.prompt_utils import SystemPromptConfig

    if isinstance(value, SystemPromptConfig):
        return value
    raise TypeError("task.system_prompt must be a string, messages list, or null.")
