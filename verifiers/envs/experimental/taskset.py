from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import wraps
from typing import Any, Callable, cast

from datasets import Dataset

from verifiers.decorators import discover_decorated
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, RolloutInput, UserMessage
from verifiers.utils.message_utils import normalize_messages

from .channels import Channel, ChannelMap
from .task import Task

DatasetSource = Dataset | Iterable[Mapping[str, Any]] | None
DatasetGetter = Callable[["Taskset"], DatasetSource]


class Taskset:
    """Dataset-shaped task collection with optional channel contributions."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "get_dataset" in cls.__dict__:
            cls.get_dataset = cached_dataset_getter(  # type: ignore[method-assign]
                cls.__dict__["get_dataset"],
                "_dataset",
                "_dataset_loaded",
            )
        if "get_eval_dataset" in cls.__dict__:
            cls.get_eval_dataset = cached_dataset_getter(  # type: ignore[method-assign]
                cls.__dict__["get_eval_dataset"],
                "_eval_dataset",
                "_eval_dataset_loaded",
            )

    def __init__(
        self,
        dataset: DatasetSource = None,
        eval_dataset: DatasetSource = None,
        rubric: Rubric | None = None,
        name: str | None = None,
    ):
        self._dataset_source = dataset
        self._eval_dataset_source = eval_dataset
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None
        self._dataset_loaded = False
        self._eval_dataset_loaded = False
        self.rubric = rubric
        self.name = name or ""
        self._stop_conditions = discover_decorated(self, "stop")
        self._cleanup_handlers = discover_decorated(self, "cleanup")
        self._teardown_handlers = discover_decorated(self, "teardown")

    def _coerce_dataset(self, dataset: DatasetSource) -> Dataset | None:
        if dataset is None or isinstance(dataset, Dataset):
            return dataset
        return Dataset.from_list([dict(row) for row in dataset])

    def channels(self, task: Task | None = None) -> ChannelMap:
        channels: dict[str, object] = {}
        if self.rubric is not None:
            channels["rubric"] = self.rubric
        if self._stop_conditions:
            channels["stop"] = self._stop_conditions
        if self._cleanup_handlers:
            channels["cleanup"] = self._cleanup_handlers
        if self._teardown_handlers:
            channels["teardown"] = self._teardown_handlers
        if task is not None:
            channels.update(task.channels)
        return channels

    def channel_objects(self) -> dict[str, object]:
        return {}

    def channel_definitions(self) -> dict[str, Channel]:
        return {}

    def has_dataset(self) -> bool:
        return (
            self._dataset_source is not None
            or type(self).get_dataset is not Taskset.get_dataset
        )

    def has_eval_dataset(self) -> bool:
        return (
            self._eval_dataset_source is not None
            or type(self).get_eval_dataset is not Taskset.get_eval_dataset
        )

    def get_dataset(self) -> Dataset | None:
        if not self._dataset_loaded:
            self._dataset = self._format_dataset(
                self._coerce_dataset(self._dataset_source)
            )
            self._dataset_loaded = True
        return self._dataset

    def get_eval_dataset(self) -> Dataset | None:
        if not self._eval_dataset_loaded:
            self._eval_dataset = self._format_dataset(
                self._coerce_dataset(self._eval_dataset_source)
            )
            self._eval_dataset_loaded = True
        return self._eval_dataset

    def _format_dataset(self, dataset: Dataset | None) -> Dataset | None:
        if dataset is None:
            return None
        if "env_id" in dataset.column_names:
            dataset = dataset.remove_columns(["env_id"])

        def format_row(row: dict[str, Any], index: int) -> dict[str, Any]:
            row = dict(row)
            row.pop("env_id", None)
            row.setdefault("example_id", index)
            if "prompt" not in row:
                question = row.get("question") or row.get("instruction") or ""
                row["prompt"] = [UserMessage(content=str(question)).model_dump()]
            return row

        return dataset.map(format_row, with_indices=True)

    def to_task(self, input: RolloutInput | Mapping[str, Any]) -> Task:
        row = dict(input)
        info = row.get("info") or {}
        if isinstance(info, str):
            info = json.loads(info)
        raw_prompt = row.get("prompt")
        if raw_prompt is None:
            question = row.get("question") or row.get("instruction") or ""
            task_prompt: Messages | str = (
                [UserMessage(content=str(question))] if question else []
            )
        elif isinstance(raw_prompt, str):
            task_prompt = raw_prompt
        elif isinstance(raw_prompt, list):
            task_prompt = cast(Messages, raw_prompt)
        else:
            raise TypeError("task.prompt must be vf.Messages or str.")
        prompt = normalize_messages(task_prompt, field_name="task.prompt")
        channels = row.get("channels") or {}
        if isinstance(channels, str):
            channels = json.loads(channels)
        known = {"prompt", "example_id", "env_id", "answer", "info", "channels"}
        inputs = {key: value for key, value in row.items() if key not in known}
        return Task(
            prompt=prompt,
            example_id=int(row.get("example_id", 0)),
            answer=row.get("answer", ""),
            info=dict(info),
            inputs=inputs,
            channels=dict(channels),
        )

    def __iter__(self):
        dataset = self.get_dataset()
        if dataset is None:
            return iter(())
        return (self.to_task(row) for row in dataset)

    def __getitem__(self, index: int) -> Task:
        dataset = self.get_dataset()
        if dataset is None:
            raise IndexError("Taskset has no dataset.")
        return self.to_task(dataset[index])

    def __len__(self) -> int:
        dataset = self.get_dataset()
        return len(dataset) if dataset is not None else 0


def cached_dataset_getter(
    getter: DatasetGetter,
    dataset_attr: str,
    loaded_attr: str,
) -> Callable[[Taskset], Dataset | None]:
    @wraps(getter)
    def wrapped(self: Taskset) -> Dataset | None:
        if not getattr(self, loaded_attr):
            dataset = self._format_dataset(self._coerce_dataset(getter(self)))
            setattr(self, dataset_attr, dataset)
            setattr(self, loaded_attr, True)
        return getattr(self, dataset_attr)

    return wrapped
