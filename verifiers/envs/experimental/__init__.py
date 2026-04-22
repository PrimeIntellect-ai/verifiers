from verifiers.envs.experimental.sandbox_mixin import SandboxMixin

__all__ = [
    "SandboxMixin",
    "SandboxSpec",
    "SandboxTaskSet",
    "Task",
    "TaskSet",
    "Harness",
    "ComposableEnv",
]


def __getattr__(name: str):
    _lazy = {
        "SandboxSpec": "verifiers.envs.experimental.composable:SandboxSpec",
        "SandboxTaskSet": "verifiers.envs.experimental.composable:SandboxTaskSet",
        "Task": "verifiers.envs.experimental.composable:Task",
        "TaskSet": "verifiers.envs.experimental.composable:TaskSet",
        "Harness": "verifiers.envs.experimental.composable:Harness",
        "ComposableEnv": "verifiers.envs.experimental.composable:ComposableEnv",
    }
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name].split(":")
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
