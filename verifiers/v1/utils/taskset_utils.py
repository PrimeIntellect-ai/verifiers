import importlib
import importlib.resources as resources
import json
from collections.abc import Iterable
from importlib.abc import Traversable
from pathlib import Path
from typing import cast

from datasets import Dataset

from ..config import resolve_config_object
from ..types import ConfigData, ConfigMap, TaskLoader, Tasks


def dataset_info_with_task(task: ConfigMap) -> ConfigData:
    return {"task": json.dumps(task)}


def resolve_task_loader(field: str, ref: str | None) -> TaskLoader | None:
    if ref is None:
        return None
    loader = resolve_config_object(ref)
    if not callable(loader):
        raise TypeError(f"TasksetConfig.{field} must resolve to a callable.")
    return cast(TaskLoader, loader)


def task_data_from_loader(
    load_tasks: TaskLoader | None,
) -> list[ConfigData]:
    if load_tasks is None:
        return []
    result = cast(Tasks, load_tasks())
    return task_data_from_result(result)


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
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        candidate_path = Path(module_file).resolve().parent / dirname
        if candidate_path.is_dir() and any(candidate_path.iterdir()):
            return candidate_path
    return None


def module_package_name(module: object) -> str | None:
    if hasattr(module, "__path__"):
        return str(getattr(module, "__name__"))
    package_name = getattr(module, "__package__", None)
    return package_name if isinstance(package_name, str) and package_name else None
