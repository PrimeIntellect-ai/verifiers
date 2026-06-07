import json
from pathlib import Path
from typing import ClassVar

from datasets import load_dataset
from pydantic import TypeAdapter

import verifiers.v1 as vf
from verifiers.v1.utils.json_utils import json_data

DATA_DIR_FIELD = "data_dir"
DATA_FILE_SUFFIX = ".jsonl"
_MESSAGES_ADAPTER = TypeAdapter(vf.Messages)


class ReplayTasksetConfig(vf.TasksetConfig):
    dataset: str | None = None
    data_dir: str | None = None


class ReplayTaskset(vf.Taskset[ReplayTasksetConfig]):
    data_dir: ClassVar[str | None] = None

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        if self.config.dataset is not None:
            if self.config.data_dir is not None:
                raise ValueError(
                    "ReplayTaskset config cannot set both dataset and data_dir."
                )
            return self.hf_tasks(self.config.dataset)
        return self.local_tasks()

    def hf_tasks(self, dataset: str) -> list[vf.JsonData]:
        rows = load_dataset(dataset, split="train")
        return [self.task_record(dict(row)) for row in rows]

    def local_tasks(self) -> list[vf.JsonData]:
        data_dir = self.load_data_dir()
        if data_dir is None:
            raise FileNotFoundError(
                f"{type(self).__name__} requires dataset or data_dir."
            )
        if not data_dir.is_dir():
            raise FileNotFoundError(f"{DATA_DIR_FIELD} must be a directory: {data_dir}")
        tasks: list[vf.JsonData] = []
        for item in sorted(data_dir.iterdir(), key=lambda path: path.name):
            if not item.is_file() or not item.name.endswith(DATA_FILE_SUFFIX):
                raise ValueError(
                    f"{DATA_DIR_FIELD} accepts only {DATA_FILE_SUFFIX} files; "
                    f"found {item.name!r}."
                )
            with item.open(encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError(
                            f"{item.name}:{line_number} must contain one JSON object."
                        )
                    tasks.append(self.task_record(record))
        if not tasks:
            raise FileNotFoundError(
                f"{DATA_DIR_FIELD} must contain at least one JSONL record."
            )
        return tasks

    def load_data_dir(self) -> Path | None:
        data_dir = self.config.data_dir or self.data_dir
        return Path(data_dir).expanduser() if data_dir is not None else None

    def task_record(self, record: dict[str, object]) -> vf.JsonData:
        messages = self.messages(record)
        if not any(message.role == "assistant" for message in messages):
            raise ValueError("Replay task messages must contain an assistant message.")
        data = dict(record)
        data["messages"] = [
            json_data(message.model_dump(mode="json", exclude_none=True))
            for message in messages
        ]
        return json_data(data)

    def messages(self, record: dict[str, object]) -> vf.Messages:
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise TypeError("Replay task messages must be a list.")
        raw_messages: list[vf.JsonData] = []
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("Replay task messages must contain JSON objects.")
            raw_messages.append(json_data(message))
        return _MESSAGES_ADAPTER.validate_python(raw_messages)


def load_taskset(config: ReplayTasksetConfig) -> ReplayTaskset:
    return ReplayTaskset(config=config)
