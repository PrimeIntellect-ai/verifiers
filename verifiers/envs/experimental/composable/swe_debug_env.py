"""Deprecated SWE-specific debug environment wrappers."""

from typing import Any
from warnings import warn

from .sandbox_debug_env import DebugStep, SandboxDebugEnv, SandboxDebugRubric


class SWEDebugRubric(SandboxDebugRubric):
    """Deprecated wrapper for SandboxDebugRubric."""

    def __init__(self, **kwargs: Any):
        warn(
            "SWEDebugRubric is deprecated with the composable debug envs; use the "
            "native v1 `debug` CLI for v1 tasksets.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class SWEDebugEnv(SandboxDebugEnv):
    """Deprecated wrapper for SandboxDebugEnv."""

    def __init__(self, *args: Any, **kwargs: Any):
        warn(
            "SWEDebugEnv is deprecated; use the native v1 `debug` CLI for v1 tasksets.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "DebugStep",
    "SandboxDebugEnv",
    "SandboxDebugRubric",
    "SWEDebugEnv",
    "SWEDebugRubric",
]
