import json
from typing import cast

from datasets import load_dataset

import verifiers as vf
from verifiers.v1 import discover_sibling_dir

DATA_SUBDIR = "data"


class ReplayTasksetConfig(vf.TasksetConfig):
    dataset: str | None = None


class ReplayTaskset(vf.Taskset[ReplayTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        if self.config.dataset is not None:
            return self.hf_tasks(self.config.dataset)
        return self.local_tasks()

    def hf_tasks(self, dataset: str) -> list[vf.JsonData]:
        rows = load_dataset(dataset, split="train")
        return [replay_task_record(dict(row)) for row in rows]

    def local_tasks(self) -> list[vf.JsonData]:
        data_dir = discover_sibling_dir(type(self), DATA_SUBDIR)
        if data_dir is None:
            raise FileNotFoundError(
                f"{type(self).__name__} requires {DATA_SUBDIR}/ next to "
                f"{type(self).__module__} when dataset is not set."
            )
        tasks: list[vf.JsonData] = []
        for item in sorted(data_dir.iterdir(), key=lambda path: path.name):
            if not item.is_file() or not item.name.endswith(".json"):
                raise ValueError(
                    f"{DATA_SUBDIR}/ accepts only .json files; found {item.name!r}."
                )
            with item.open(encoding="utf-8") as f:
                record = json.load(f)
            if not isinstance(record, dict):
                raise TypeError(f"{item.name} must contain one JSON object.")
            tasks.append(replay_task_record(record))
        if not tasks:
            raise FileNotFoundError(
                f"{DATA_SUBDIR}/ must contain at least one .json file."
            )
        return tasks


def replay_task_record(record: dict[str, object]) -> vf.JsonData:
    messages = replay_messages(record)
    if not any(message.role == "assistant" for message in messages):
        raise ValueError("Replay task messages must contain an assistant message.")
    data = dict(record)
    data["messages"] = [
        cast(vf.JsonData, message.model_dump(mode="json", exclude_none=True))
        for message in messages
    ]
    return cast(vf.JsonData, data)


def replay_messages(record: dict[str, object]) -> vf.Messages:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise TypeError("Replay task messages must be a list.")
    raw_messages: list[dict[str, object]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("Replay task messages must contain JSON objects.")
        raw_messages.append(cast(dict[str, object], message))
    return vf.get_messages(raw_messages)


def load_taskset(config: ReplayTasksetConfig) -> ReplayTaskset:
    return ReplayTaskset(config=config)
