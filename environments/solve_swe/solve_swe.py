from __future__ import annotations

from typing import Any

from verifiers.envs.experimental.composable import SolveEnv
from verifiers.envs.experimental.composable.tasksets.swe import make_swe_taskset


def load_environment(task_type: str = "r2e", **solve_kwargs: Any) -> SolveEnv:
    """Gold-patch validation of any SWE taskset (no agent, no LLM).

    ``task_type`` selects the SWE backend (``r2e``, ``multiswe``,
    ``swebench``, ``openswe``). Remaining kwargs forward to
    ``SolveEnv``.
    """
    return SolveEnv(taskset=make_swe_taskset(backend=task_type), **solve_kwargs)
