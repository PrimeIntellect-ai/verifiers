from verifiers.envs.experimental.composable.task import (
    SandboxSpec,
    Task,
    TaskSet,
    SandboxTaskSet,
    discover_sibling_dir,
)
from verifiers.envs.experimental.composable.harness import Harness
from verifiers.envs.experimental.composable.composable_env import ComposableEnv

# Backward-compatible alias — RlmComposableEnv is no longer a separate class.
# All its functionality (install_env, upload dirs, metrics) is now in ComposableEnv.
RlmComposableEnv = ComposableEnv

__all__ = [
    "SandboxSpec",
    "Task",
    "TaskSet",
    "SandboxTaskSet",
    "Harness",
    "ComposableEnv",
    "RlmComposableEnv",
    "discover_sibling_dir",
]
