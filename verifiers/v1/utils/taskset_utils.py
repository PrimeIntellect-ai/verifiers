import importlib
import importlib.resources as resources
import json
from collections.abc import Callable, Iterable
from importlib.abc import Traversable
from pathlib import Path
from typing import cast
from ..types import ConfigData, ConfigMap


def dataset_info_with_task(task: ConfigMap) -> ConfigData:
    return {"task": json.dumps(task)}


def rows_from_source(
    source: Iterable[ConfigMap] | Callable[[], Iterable[ConfigMap]] | None,
) -> list[ConfigData]:
    if source is None:
        return []
    if callable(source):
        source_loader = cast(Callable[[], Iterable[ConfigMap]], source)
        return [dict(row) for row in source_loader()]
    return [dict(row) for row in source]


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
