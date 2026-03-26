from verifiers.envs.experimental.sandbox_mixin import SandboxMixin

__all__ = [
    "SandboxMixin",
    # Composable architecture
    "Task",
    "TaskSet",
    "MergedTaskSet",
    "ComposableEnv",
]


def __getattr__(name: str):
    """Lazy imports for composable architecture classes."""
    _lazy = {
        "Task": "verifiers.envs.experimental.task:Task",
        "TaskSet": "verifiers.envs.experimental.task:TaskSet",
        "MergedTaskSet": "verifiers.envs.experimental.task:MergedTaskSet",
        "ComposableEnv": "verifiers.envs.experimental.composable_env:ComposableEnv",
    }
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name].split(":")
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
