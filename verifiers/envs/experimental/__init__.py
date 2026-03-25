from verifiers.envs.experimental.sandbox_mixin import SandboxMixin

__all__ = [
    "SandboxMixin",
    # Composable architecture (Task / Agent / Environment)
    "Task",
    "TaskSet",
    "MergedTaskSet",
    "Agent",
    "ReActAgent",
    "LLMAgent",  # alias for ReActAgent
    "SingleTurnAgent",
    "BinaryAgent",
    "ComposableEnv",
    "UserSimEnv",
    "SweTaskAdapter",
    "HarborTaskSet",
    "HarborTask",
]


def __getattr__(name: str):
    """Lazy imports for composable architecture classes."""
    _lazy = {
        "Task": "verifiers.envs.experimental.task:Task",
        "TaskSet": "verifiers.envs.experimental.task:TaskSet",
        "MergedTaskSet": "verifiers.envs.experimental.task:MergedTaskSet",
        "Agent": "verifiers.envs.experimental.agent:Agent",
        "ReActAgent": "verifiers.envs.experimental.agent:ReActAgent",
        "LLMAgent": "verifiers.envs.experimental.agent:LLMAgent",
        "SingleTurnAgent": "verifiers.envs.experimental.agent:SingleTurnAgent",
        "BinaryAgent": "verifiers.envs.experimental.binary_agent:BinaryAgent",
        "ComposableEnv": "verifiers.envs.experimental.composable_env:ComposableEnv",
        "UserSimEnv": "verifiers.envs.experimental.user_sim_env:UserSimEnv",
        "SweTaskAdapter": "verifiers.envs.experimental.swe_task_adapter:SweTaskAdapter",
        "HarborTaskSet": "verifiers.envs.experimental.harbor_task_adapter:HarborTaskSet",
        "HarborTask": "verifiers.envs.experimental.harbor_task_adapter:HarborTask",
    }
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name].split(":")
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
