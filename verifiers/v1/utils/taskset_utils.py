import importlib
import importlib.resources as resources
import inspect
import json
from collections.abc import Iterable
from importlib.abc import Traversable
from pathlib import Path
from typing import cast

from datasets import Dataset
from pydantic import BaseModel

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
    config: object | None = None,
) -> list[ConfigData]:
    if load_tasks is None:
        return []
    result = call_task_loader(load_tasks, config)
    return task_data_from_result(result)


def call_task_loader(
    load_tasks: TaskLoader,
    config: object | None,
) -> Tasks:
    task_args = task_config_args(config)
    sig = inspect.signature(load_tasks)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return load_tasks(**task_args)
    keyword_names = {
        name
        for name, parameter in sig.parameters.items()
        if parameter.kind
        not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL)
    }
    allowed = {key: value for key, value in task_args.items() if key in keyword_names}
    return load_tasks(**allowed)


def task_data_from_result(result: Tasks) -> list[ConfigData]:
    if isinstance(result, Dataset):
        return [dict(row) for row in result]
    if isinstance(result, Iterable):
        rows = cast(Iterable[ConfigMap], result)
        return [dict(row) for row in rows]
    raise TypeError(
        "Task loader must return a datasets.Dataset or an iterable of mappings."
    )


def task_config_args(config: object | None) -> ConfigData:
    if config is None:
        return {}
    if isinstance(config, BaseModel):
        data = config.model_dump(mode="python")
    elif isinstance(config, dict):
        data = dict(config)
    else:
        return {"config": config, "taskset_config": config}
    return {**data, "config": config, "taskset_config": config}


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
