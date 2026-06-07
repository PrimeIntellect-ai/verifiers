import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic import TypeAdapter, model_validator
from typing_extensions import Self
from verifiers.types import Messages, SystemMessage

from ..config import Config
from ..types import JsonData, PromptInput
from .config_utils import current_config_ref_module

if TYPE_CHECKING:
    from ..task import Task

_MESSAGES_ADAPTER = TypeAdapter(Messages)


SystemPromptStrategy = Literal["REJECT", "TH", "HT", "T", "H", "T_OR_H", "H_OR_T"]
SystemPromptTasksetSource = Literal["task", "taskset"]


@dataclass(frozen=True)
class SystemPromptResolution:
    harness: list[JsonData]
    taskset: list[JsonData]
    taskset_source: SystemPromptTasksetSource | None

    def apply_strategy(self, strategy: SystemPromptStrategy) -> list[JsonData]:
        harness = [dict(message) for message in self.harness]
        taskset = [dict(message) for message in self.taskset]
        if strategy == "HT":
            return [*harness, *taskset]
        if strategy == "TH":
            return [*taskset, *harness]
        if strategy == "REJECT":
            if self.harness and self.taskset:
                raise ValueError(
                    "Multiple system_prompt sides cannot be resolved: "
                    f"harness, {self.taskset_source or 'taskset'}. "
                    "Set system_prompt_strategy='HT', 'TH', 'H', 'T', "
                    "'H_OR_T', or 'T_OR_H'."
                )
            return [*harness, *taskset]
        if strategy == "H_OR_T":
            return harness or taskset
        if strategy == "T_OR_H":
            return taskset or harness
        if strategy == "H":
            return harness
        if strategy == "T":
            return taskset
        raise ValueError(
            "system_prompt_strategy must be one of REJECT, TH, HT, T, H, "
            "T_OR_H, H_OR_T."
        )


class SystemPromptConfig(Config):
    path: str | None = None
    messages: list[JsonData] = []

    @model_validator(mode="after")
    def validate_one_input(self) -> Self:
        inputs = [
            self.path is not None,
            bool(self.messages),
        ]
        if sum(inputs) != 1:
            raise ValueError(
                "SystemPromptConfig requires exactly one of path or messages."
            )
        return self

    def load(self, field_name: str) -> PromptInput | None:
        if self.path is not None:
            return read_system_prompt_path(
                resolve_system_prompt_path(self.path), field_name=field_name
            )
        return self.messages


SystemPrompt: TypeAlias = PromptInput | SystemPromptConfig | None


def normalize_system_prompt(
    value: SystemPrompt,
    field_name: str = "system_prompt",
) -> list[JsonData]:
    value = resolve_system_prompt_input(value, field_name=field_name)
    if value is None:
        return []
    if isinstance(value, str):
        return [SystemMessage(content=value).model_dump(exclude_none=True)]
    messages = _MESSAGES_ADAPTER.validate_python(value)
    for message in messages:
        if getattr(message, "role", None) != "system":
            raise ValueError(f"{field_name} accepts only system messages.")
    return dump_messages(messages)


def resolve_system_prompt_input(
    value: SystemPrompt,
    *,
    field_name: str,
) -> PromptInput | None:
    if value is None:
        return None
    if isinstance(value, SystemPromptConfig):
        return value.load(field_name)
    return value


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
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{field_name} path {str(path)!r} could not be read.") from exc
    if not text:
        raise ValueError(f"{field_name} path {str(path)!r} must not be empty.")
    return text


def resolve_system_prompt(
    *,
    task: "Task",
    taskset_system_prompt: list[JsonData],
    harness_system_prompt: list[JsonData],
    strategy: SystemPromptStrategy,
) -> list[JsonData]:
    return system_prompt_resolution(
        task=task,
        taskset_system_prompt=taskset_system_prompt,
        harness_system_prompt=harness_system_prompt,
    ).apply_strategy(strategy)


def system_prompt_resolution(
    *,
    task: "Task",
    taskset_system_prompt: list[JsonData],
    harness_system_prompt: list[JsonData],
) -> SystemPromptResolution:
    task_system_prompt = normalize_system_prompt(
        task.system_prompt,
        field_name="task.system_prompt",
    )
    return SystemPromptResolution(
        harness=[dict(message) for message in harness_system_prompt],
        taskset=(
            [dict(message) for message in task_system_prompt]
            if task_system_prompt
            else [dict(message) for message in taskset_system_prompt]
        ),
        taskset_source=(
            "task"
            if task_system_prompt
            else "taskset"
            if taskset_system_prompt
            else None
        ),
    )


def dump_messages(messages: Messages) -> list[JsonData]:
    return [message.model_dump(exclude_none=True) for message in messages]
