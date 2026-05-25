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
from ..types import ConfigData, ConfigMap, Handler, TaskLoader, Tasks


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
    result = cast(Tasks, call_loader_with_config(load_tasks, config, "Task loader"))
    return task_data_from_result(result)


def call_loader_with_config(
    loader: Handler,
    config: object | None,
    context: str = "Loader",
) -> object:
    task_args = task_config_args(config)
    sig = inspect.signature(loader)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return loader(**task_args)
    keyword_names = {
        name
        for name, parameter in sig.parameters.items()
        if parameter.kind
        not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL)
    }
    missing = [
        name
        for name, parameter in sig.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in task_args
    ]
    if missing:
        loader_name = getattr(loader, "__name__", type(loader).__name__)
        raise TypeError(
            f"{context} {loader_name!r} requires config field(s): "
            f"{', '.join(repr(name) for name in missing)}."
        )
    allowed = {key: value for key, value in task_args.items() if key in keyword_names}
    return loader(**allowed)


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
