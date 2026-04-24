__all__ = [
    "DatasetTaskset",
    "HarborRubric",
    "HarborTaskset",
]

_LAZY_IMPORTS = {
    "DatasetTaskset": (
        "verifiers.envs.experimental.modules.tasksets.dataset_taskset:DatasetTaskset"
    ),
    "HarborTaskset": (
        "verifiers.envs.experimental.modules.tasksets.harbor_taskset:HarborTaskset"
    ),
    "HarborRubric": (
        "verifiers.envs.experimental.modules.tasksets.harbor_taskset:HarborRubric"
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_path, attr = _LAZY_IMPORTS[name].split(":")
    return getattr(importlib.import_module(module_path), attr)
