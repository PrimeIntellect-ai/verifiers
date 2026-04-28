from __future__ import annotations

import importlib
from typing import Any

from verifiers.envs.experimental.composable import SolveEnv


def load_environment(
    taskset: str,
    taskset_args: dict | None = None,
    **solve_kwargs: Any,
) -> SolveEnv:
    """``vf-eval`` factory for ``SolveEnv``.

    ``taskset`` is a ``"module.path:attr"`` spec — ``attr`` may be a
    factory callable or a ``SandboxTaskSet`` subclass. ``taskset_args``
    is forwarded to it; remaining kwargs go to ``SolveEnv``.
    """
    module_path, sep, attr = taskset.partition(":")
    if not sep or not attr:
        raise ValueError(f"taskset spec must be 'module.path:attr', got: {taskset!r}")
    obj = getattr(importlib.import_module(module_path), attr)
    return SolveEnv(taskset=obj(**(taskset_args or {})), **solve_kwargs)
