from __future__ import annotations

import uuid
from collections.abc import Mapping
from copy import deepcopy
from typing import TypeVar, cast

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from verifiers.types import Messages
from verifiers.utils.message_utils import normalize_messages

from .types import JsonData
from .utils.task_freeze_utils import assert_serializable


class Task(BaseModel, extra="forbid", frozen=True):
    """Immutable serializable task specification. Subclass for task-specific data."""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    row_id: int = 0
    prompt: Messages = Field(default_factory=list)
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
        assert_serializable(self.model_dump(mode="json", exclude_none=True))
        return self

    def to_record(self) -> JsonData:
        return cast(JsonData, self.model_dump(mode="json", exclude_none=True))


TaskT = TypeVar("TaskT", bound=Task)


def normalized_task_prompt(value: object) -> Messages:
    messages = normalize_messages(cast(Messages, value or []), field_name="task.prompt")
    for message in messages:
        if getattr(message, "role", None) == "system":
            raise ValueError("task.prompt must not contain system messages.")
    return messages
