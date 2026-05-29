import importlib
import importlib.resources as resources
import json
import uuid
from collections.abc import Iterable, Mapping
from copy import deepcopy
from importlib.abc import Traversable
from pathlib import Path
from typing import cast

from datasets import Dataset
from verifiers.types import task_payload_from_info

from ..task import Task
from ..types import ConfigData, ConfigMap, TaskRow, Tasks


def task_from_row(row: TaskRow | Task, taskset_id: str) -> Task:
    if not isinstance(row, Mapping):
        raise TypeError("Taskset.to_task expects a task row.")
    serialized_task = task_payload_from_info(row.get("info"))
    if serialized_task is not None:
        row = serialized_task
    data = deepcopy(dict(row))
    if "prompt" not in data:
        question = data.get("question")
        data["prompt"] = (
            [{"role": "user", "content": str(question)}] if question is not None else []
        )
    task = Task(data)
    task["taskset_id"] = taskset_id
    if "task_id" in task:
        task["task_id"] = str(task["task_id"])
    elif "example_id" in task:
        task["task_id"] = str(task["example_id"])
    else:
        task["task_id"] = uuid.uuid4().hex
    return task.freeze()


def dataset_rows_from_tasks(
    rows: Iterable[TaskRow], taskset_id: str
) -> list[ConfigData]:
    dataset_rows: list[ConfigData] = []
    for index, row in enumerate(rows):
        normalized = deepcopy(dict(row))
        normalized.setdefault("example_id", index)
        task_payload = dict(task_from_row(normalized, taskset_id))
        dataset_row: ConfigData = {
            "prompt": task_payload["prompt"],
            "example_id": normalized["example_id"],
            "info": {"task": json.dumps(task_payload)},
        }
        if "answer" in normalized:
            dataset_row["answer"] = normalized["answer"]
        dataset_rows.append(dataset_row)
    return dataset_rows


def task_data_from_result(result: Tasks) -> list[ConfigData]:
    if isinstance(result, Dataset):
        return [dict(row) for row in result]
    if isinstance(result, Iterable):
        rows = cast(Iterable[ConfigMap], result)
        return [dict(row) for row in rows]
    raise TypeError(
        "Task loader must return a datasets.Dataset or an iterable of mappings."
    )


def discover_sibling_dir(
    taskset_cls: type[object], dirname: str
) -> Traversable | Path | None:
    module = importlib.import_module(taskset_cls.__module__)
    package_name = module_package_name(module)
    if package_name is not None:
        try:
            candidate = resources.files(package_name) / dirname
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
        except (
            FileNotFoundError,
            ModuleNotFoundError,
            NotADirectoryError,
            TypeError,
            ValueError,
        ):
            pass
    module_file = module.__dict__.get("__file__")
    if isinstance(module_file, str):
        candidate_path = Path(module_file).resolve().parent / dirname
        if candidate_path.is_dir() and any(candidate_path.iterdir()):
            return candidate_path
    return None


def module_package_name(module: object) -> str | None:
    module_attrs = module.__dict__
    if "__path__" in module_attrs:
        return str(module_attrs["__name__"])
    package_name = module_attrs.get("__package__")
    return package_name if isinstance(package_name, str) and package_name else None
