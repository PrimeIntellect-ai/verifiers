# ruff: noqa

from verifiers.legacy.envs.experimental.sandbox_mixin import SandboxMixin

__all__ = [
    "SandboxMixin",
    "SandboxSpec",
    "SandboxTaskSet",
    "Task",
    "TaskSet",
    "Harness",
    "ComposableEnv",
    "SandboxDebugEnv",
    "SandboxDebugRubric",
    "SWEDebugEnv",
]


def __getattr__(name: str):
    _lazy = {
        "SandboxSpec": "verifiers.legacy.envs.experimental.composable:SandboxSpec",
        "SandboxTaskSet": "verifiers.legacy.envs.experimental.composable:SandboxTaskSet",
        "Task": "verifiers.legacy.envs.experimental.composable:Task",
        "TaskSet": "verifiers.legacy.envs.experimental.composable:TaskSet",
        "Harness": "verifiers.legacy.envs.experimental.composable:Harness",
        "ComposableEnv": "verifiers.legacy.envs.experimental.composable:ComposableEnv",
        "SandboxDebugEnv": "verifiers.legacy.envs.experimental.composable:SandboxDebugEnv",
        "SandboxDebugRubric": "verifiers.legacy.envs.experimental.composable:SandboxDebugRubric",
        "SWEDebugEnv": "verifiers.legacy.envs.experimental.composable:SWEDebugEnv",
    }
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name].split(":")
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
