import importlib
import importlib.resources as resources
from collections.abc import Iterable
from contextlib import suppress
from importlib.abc import Traversable
from pathlib import Path

from datasets import Dataset

from ..task import Task
from ..types import JsonData, Tasks
from .json_utils import json_data


def task_from_dataset_record(record: JsonData, task_type: type[Task] = Task) -> Task:
    return prepare_task(task_type.model_validate(json_data(record)), task_type)


def prepare_task(task: Task, task_type: type[Task] = Task) -> Task:
    if not isinstance(task, Task):
        raise TypeError("v1 task loaders must return Task objects.")
    if isinstance(task, task_type):
        return task
    data = json_data(task.model_dump(mode="json", exclude_none=True))
    return task_type.model_validate(data)


def dataset_record_from_task(
    task: Task,
    index: int,
) -> JsonData:
    data = json_data(task.model_dump(mode="json", exclude_none=True))
    data["row_id"] = index
    normalized = prepare_task(type(task).model_validate(data), type(task))
    task_payload = json_data(normalized.model_dump(mode="json", exclude_none=True))
    task_payload["example_id"] = index
    return task_payload


def dataset_records_from_tasks(tasks: Iterable[Task]) -> list[JsonData]:
    dataset_records: list[JsonData] = []
    for index, task in enumerate(tasks):
        dataset_records.append(dataset_record_from_task(task, index))
    return dataset_records


def dataset_from_result(result: Tasks) -> Dataset:
    return dataset_from_result_typed(result, Task)


def dataset_from_result_typed(result: Tasks, task_type: type[Task]) -> Dataset:
    if isinstance(result, Dataset):
        records: list[JsonData] = []
        for index, record in enumerate(result):
            row = json_data(dict(record))
            row["example_id"] = index
            task = task_from_dataset_record(row, task_type)
            records.append(dataset_record_from_task(task, index))
        return Dataset.from_list(records)
    tasks = tasks_from_result_typed(result, task_type)
    return Dataset.from_list(dataset_records_from_tasks(tasks))


def tasks_from_result(result: Tasks) -> list[Task]:
    return tasks_from_result_typed(result, Task)


def tasks_from_result_typed(result: Tasks, task_type: type[Task]) -> list[Task]:
    if isinstance(result, Dataset):
        return [
            task_from_dataset_record(json_data(dict(record)), task_type)
            for record in result
        ]
    if isinstance(result, Iterable):
        tasks: list[Task] = []
        for item in result:
            if isinstance(item, Task):
                tasks.append(prepare_task(item, task_type))
            elif isinstance(item, dict):
                tasks.append(task_from_dataset_record(json_data(item), task_type))
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
