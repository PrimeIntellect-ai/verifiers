from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy

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


class Resources(BaseModel, extra="forbid", frozen=True):
    cpu_cores: float | None = None
    memory_gb: float | None = None
    gpu_count: int | None = None
    disk_gb: float | None = None


class Task(BaseModel, extra="forbid", frozen=True):
    """Immutable serializable task specification. Subclass for task-specific data."""

    task_id: str = ""
    row_id: int = 0
    prompt: Messages = Field(default_factory=list)
    system_prompt: SystemPrompt = None
    toolsets: TaskVisibility | None = None
    tools: TaskVisibility | None = None
    user: bool | None = None
    name: str | None = None
    description: str | None = None
    image: str | None = None
    resources: Resources = Field(default_factory=Resources)
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
                raw["prompt"] = cls.normalize_prompt(raw["prompt"])
            if "system_prompt" in raw:
                raw["system_prompt"] = cls.normalize_system_prompt(raw["system_prompt"])
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
        object.__setattr__(self, "prompt", type(self).normalize_prompt(self.prompt))
        object.__setattr__(
            self,
            "system_prompt",
            type(self).normalize_system_prompt(self.system_prompt),
        )
        if not self.task_id:
            object.__setattr__(self, "task_id", self.default_task_id())
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
        serialized = {str(key): value for key, value in data.items()}
        if info.mode != "json":
            return serialized
        if "prompt" in serialized:
            serialized["prompt"] = dump_messages(self.prompt)
        if "system_prompt" in serialized:
            if self.system_prompt:
                serialized["system_prompt"] = list(
                    type(self).normalize_system_prompt(self.system_prompt)
                )
            else:
                serialized.pop("system_prompt", None)
        return serialized

    @classmethod
    def normalize_prompt(cls, value: object) -> Messages:
        messages: Messages
        if isinstance(value, str):
            messages = [UserMessage(content=value)]
        else:
            messages = MESSAGES_ADAPTER.validate_python(value or [])
        for message in messages:
            if getattr(message, "role", None) == "system":
                raise ValueError("task.prompt must not contain system messages.")
        return messages

    @classmethod
    def normalize_system_prompt(cls, value: object) -> list[JsonData]:
        return normalize_system_prompt(
            cls.system_prompt_input(value),
            field_name="task.system_prompt",
        )

    @classmethod
    def system_prompt_input(cls, value: object) -> SystemPrompt:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return MESSAGES_ADAPTER.validate_python(value)
        from .utils.prompt_utils import SystemPromptConfig

        if isinstance(value, Mapping):
            return SystemPromptConfig.model_validate(dict(value))
        if isinstance(value, SystemPromptConfig):
            return value
        raise TypeError("task.system_prompt must be a string, messages list, or null.")

    def default_task_id(self) -> str:
        data = self.model_dump(
            mode="json",
            exclude={"task_id"},
            exclude_none=True,
            exclude_defaults=True,
        )
        payload = {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "task": data,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:24]
