from verifiers.envs.experimental.sandbox_mixin import SandboxMixin

__all__ = [
    "Channel",
    "ChannelConfig",
    "ChannelContext",
    "ChannelMap",
    "DatasetTaskset",
    "Env",
    "HarborRubric",
    "HarborTaskset",
    "Harness",
    "Resources",
    "SandboxMixin",
    "SandboxSeed",
    "SandboxSpec",
    "SandboxTimeouts",
    "Taskset",
    "SandboxTaskSet",
    "Task",
    "TaskSet",
    "CallableTool",
    "MCPServerSpec",
    "ToolRegistry",
    "User",
    "ComposableEnv",
]


def __getattr__(name: str):
    _lazy = {
        "Channel": "verifiers.envs.experimental.channels:Channel",
        "ChannelConfig": "verifiers.envs.experimental.channels:ChannelConfig",
        "ChannelContext": "verifiers.envs.experimental.channels:ChannelContext",
        "ChannelMap": "verifiers.envs.experimental.channels:ChannelMap",
        "DatasetTaskset": "verifiers.envs.experimental.modules.tasksets:DatasetTaskset",
        "Env": "verifiers.envs.experimental.env:Env",
        "HarborRubric": "verifiers.envs.experimental.modules.tasksets:HarborRubric",
        "HarborTaskset": "verifiers.envs.experimental.modules.tasksets:HarborTaskset",
        "Harness": "verifiers.envs.experimental.harness:Harness",
        "Resources": "verifiers.envs.experimental.resources:Resources",
        "SandboxSeed": "verifiers.envs.experimental.channels:SandboxSeed",
        "SandboxTimeouts": "verifiers.envs.experimental.channels:SandboxTimeouts",
        "Taskset": "verifiers.envs.experimental.taskset:Taskset",
        "ToolRegistry": "verifiers.envs.experimental.channels:ToolRegistry",
        "User": "verifiers.envs.experimental.channels:User",
        "SandboxSpec": "verifiers.envs.experimental.channels:SandboxSpec",
        "SandboxTaskSet": "verifiers.envs.experimental.composable:SandboxTaskSet",
        "Task": "verifiers.envs.experimental.task:Task",
        "TaskSet": "verifiers.envs.experimental.composable:TaskSet",
        "CallableTool": "verifiers.envs.experimental.channels:CallableTool",
        "MCPServerSpec": "verifiers.envs.experimental.channels:MCPServerSpec",
        "ComposableEnv": "verifiers.envs.experimental.composable:ComposableEnv",
    }
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name].split(":")
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
