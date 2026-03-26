from verifiers.envs.experimental.sandbox_mixin import SandboxMixin

__all__ = [
    "SandboxMixin",
    "TaskSpec",
    "TaskSet",
    "ComposableEnv",
]


def __getattr__(name: str):
    _lazy = {
        "TaskSpec": "verifiers.envs.experimental.task:TaskSpec",
        "TaskSet": "verifiers.envs.experimental.task:TaskSet",
        "ComposableEnv": "verifiers.envs.experimental.composable_env:ComposableEnv",
    }
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name].split(":")
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
