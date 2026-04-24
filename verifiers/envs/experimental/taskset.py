from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

from datasets import Dataset

from verifiers.decorators import discover_decorated
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, RolloutInput, UserMessage
from verifiers.utils.message_utils import normalize_messages

from .channels import Channel, ChannelMap
from .task import Task

LoadedSource = Dataset | Iterable[Mapping[str, Any]] | None
Source = LoadedSource | Callable[[], LoadedSource]


class Taskset:
    """Dataset-shaped task collection with optional channel contributions."""

    def __init__(
        self,
        source: Source = None,
        eval_source: Source = None,
        rubric: Rubric | None = None,
        tools: Iterable[object] | None = None,
        name: str | None = None,
    ):
        self._source = source
        self._eval_source = eval_source
        self._dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None
        self._dataset_loaded = False
        self._eval_dataset_loaded = False
        self.rubric = rubric
        self.tools = list(tools or [])
        self.name = name or ""
        self._stop_conditions = discover_decorated(self, "stop")
        self._cleanup_handlers = discover_decorated(self, "cleanup")
        self._teardown_handlers = discover_decorated(self, "teardown")

    def _load_source(self, source: Source) -> Dataset | None:
        loaded = source() if callable(source) else source
        return self._coerce_dataset(loaded)

    def _coerce_dataset(self, dataset: LoadedSource) -> Dataset | None:
        if dataset is None or isinstance(dataset, Dataset):
            return dataset
        return Dataset.from_list([dict(row) for row in dataset])

    def channels(self, task: Task | None = None) -> ChannelMap:
        channels: dict[str, object] = {}
        if self.rubric is not None:
            channels["rubric"] = self.rubric
        if self.tools:
            channels["tools"] = self.tools
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
        return self._source is not None

    def has_eval_dataset(self) -> bool:
        return self._eval_source is not None

    def get_dataset(self) -> Dataset | None:
        if not self._dataset_loaded:
            self._dataset = self._format_dataset(self._load_source(self._source))
            self._dataset_loaded = True
        return self._dataset

    def get_eval_dataset(self) -> Dataset | None:
        if not self._eval_dataset_loaded:
            self._eval_dataset = self._format_dataset(
                self._load_source(self._eval_source)
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
