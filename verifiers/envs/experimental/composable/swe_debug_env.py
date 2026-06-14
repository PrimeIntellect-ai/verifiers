"""Deprecated SWE-specific debug environment wrappers."""

from typing import Any
from warnings import warn

from .sandbox_debug_env import DebugStep, SandboxDebugEnv, SandboxDebugRubric


class SWEDebugRubric(SandboxDebugRubric):
    """Deprecated wrapper for SandboxDebugRubric."""

    def __init__(self, **kwargs: Any):
        warn(
            "SWEDebugRubric is deprecated; use SandboxDebugRubric.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class SWEDebugEnv(SandboxDebugEnv):
    """Deprecated wrapper for SandboxDebugEnv."""

    def __init__(self, *args: Any, **kwargs: Any):
        warn(
            "SWEDebugEnv is deprecated; use SandboxDebugEnv.",
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
