import importlib
import importlib.resources as resources
import json
import uuid
from collections.abc import Iterable
from contextlib import suppress
from copy import deepcopy
from importlib.abc import Traversable
from pathlib import Path
from typing import cast

from datasets import Dataset
from verifiers.types import task_payload_from_info

from ..task import Task
from ..types import JsonData, Tasks
from .serialization_utils import serializable


def task_from_dataset_record(record: JsonData, taskset_id: str) -> Task:
    record_data = serializable(record)
    assert isinstance(record_data, dict)
    record_json = cast(JsonData, record_data)
    serialized_task = task_payload_from_info(record_json.get("info"))
    if serialized_task is not None:
        record_json = serialized_task
    data = deepcopy(dict(record_json))
    if "prompt" not in data:
        question = data.get("question")
        data["prompt"] = (
            [{"role": "user", "content": str(question)}] if question is not None else []
        )
    return prepare_task(Task(cast(JsonData, data)), taskset_id)


def prepare_task(task: Task, taskset_id: str) -> Task:
    if not isinstance(task, Task):
        raise TypeError("v1 task loaders must return Task objects.")
    prepared = Task(cast(JsonData, dict(task)))
    prepared["taskset_id"] = taskset_id
    if prepared.get("task_id") is not None:
        prepared["task_id"] = str(prepared["task_id"])
    else:
        prepared["task_id"] = uuid.uuid4().hex
    return prepared.freeze()


def dataset_record_from_task(
    task: Task,
    taskset_id: str,
    index: int,
    record: JsonData | None = None,
) -> JsonData:
    data = Task(cast(JsonData, dict(task)))
    data["example_id"] = index
    normalized = prepare_task(data, taskset_id)
    task_payload = dict(normalized)
    dataset_record = deepcopy(dict(record or {}))
    dataset_record["prompt"] = task_payload["prompt"]
    dataset_record["example_id"] = task_payload["example_id"]
    info = dataset_record.get("info")
    if not isinstance(info, dict):
        info = {}
    dataset_record["info"] = {**info, "task": json.dumps(task_payload)}
    if "answer" in normalized:
        dataset_record["answer"] = normalized["answer"]
    return cast(JsonData, dataset_record)


def dataset_records_from_tasks(
    tasks: Iterable[Task], taskset_id: str
) -> list[JsonData]:
    dataset_records: list[JsonData] = []
    for index, task in enumerate(tasks):
        dataset_records.append(dataset_record_from_task(task, taskset_id, index))
    return dataset_records


def dataset_from_result(result: Tasks, taskset_id: str) -> Dataset:
    if isinstance(result, Dataset):
        records: list[JsonData] = []
        for index, record in enumerate(result):
            row = cast(JsonData, dict(record))
            row["example_id"] = index
            task = task_from_dataset_record(row, taskset_id)
            records.append(dataset_record_from_task(task, taskset_id, index, row))
        return Dataset.from_list(records)
    tasks = tasks_from_result(result, taskset_id)
    return Dataset.from_list(dataset_records_from_tasks(tasks, taskset_id))


def tasks_from_result(result: Tasks, taskset_id: str) -> list[Task]:
    if isinstance(result, Dataset):
        return [
            task_from_dataset_record(cast(JsonData, dict(record)), taskset_id)
            for record in result
        ]
    if isinstance(result, Iterable):
        tasks: list[Task] = []
        for item in result:
            if isinstance(item, Task):
                tasks.append(prepare_task(item, taskset_id))
            elif isinstance(item, dict):
                tasks.append(task_from_dataset_record(cast(JsonData, item), taskset_id))
            else:
                raise TypeError(
                    "Task loader iterables must contain Task objects or JSON task "
                    "records."
                )
        return tasks
    raise TypeError("Task loader must return a Dataset or an iterable of tasks.")


def discover_sibling_dir(
    taskset_cls: type[object], dirname: str, *, require_non_empty: bool = False
) -> Traversable | Path | None:
    module = importlib.import_module(taskset_cls.__module__)
    package_name = module_package_name(module)
    if package_name is not None:
        with suppress(
            FileNotFoundError,
            ModuleNotFoundError,
            NotADirectoryError,
            TypeError,
            ValueError,
        ):
            candidate = resources.files(package_name) / dirname
            if sibling_dir_matches(candidate, require_non_empty=require_non_empty):
                return candidate
    module_file = module.__dict__.get("__file__")
    if isinstance(module_file, str):
        candidate_path = Path(module_file).resolve().parent / dirname
        if sibling_dir_matches(candidate_path, require_non_empty=require_non_empty):
            return candidate_path
    return None


def sibling_dir_matches(
    candidate: Traversable | Path, *, require_non_empty: bool
) -> bool:
    if not candidate.is_dir():
        return False
    if not require_non_empty:
        return True
    return any(candidate.iterdir())


def module_package_name(module: object) -> str | None:
    module_attrs = module.__dict__
    if "__path__" in module_attrs:
        return str(module_attrs["__name__"])
    package_name = module_attrs.get("__package__")
    return package_name if isinstance(package_name, str) and package_name else None
