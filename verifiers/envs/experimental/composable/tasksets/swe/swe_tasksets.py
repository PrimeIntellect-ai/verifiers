"""SWE TaskSet factories.

from verifiers.envs.experimental.composable.tasksets.swe import make_r2e_taskset, make_swebench_taskset

r2e = make_r2e_taskset()
bench = make_swebench_taskset()
"""

from __future__ import annotations

from typing import Any

from verifiers.envs.experimental.composable import TaskSet


def make_swe_taskset(
    backend: str = "r2e",
    **kwargs: Any,
) -> TaskSet:
    """Create a SWE TaskSet from a backend name."""
    factories = {
        "r2e": make_r2e_taskset,
        "swebench": make_swebench_taskset,
        "openswe": make_openswe_taskset,
        "multiswe": make_multiswe_taskset,
        "swelego-real": make_swelego_real_taskset,
    }
    if backend not in factories:
        raise ValueError(
            f"Unknown SWE backend: {backend!r}. Available: {list(factories)}"
        )
    return factories[backend](**kwargs)


def make_r2e_taskset(**kwargs: Any) -> TaskSet:
    """R2E-Gym TaskSet (default: 4578 instances)."""
    from verifiers.envs.experimental.composable.tasksets.swe.r2e_gym import (
        R2EGymTaskSet,
    )

    return R2EGymTaskSet(**kwargs)


def make_swebench_taskset(**kwargs: Any) -> TaskSet:
    """SWE-bench Verified TaskSet."""
    from verifiers.envs.experimental.composable.tasksets.swe.swe_bench import (
        SWEBenchTaskSet,
    )

    return SWEBenchTaskSet(**kwargs)


def make_multiswe_taskset(**kwargs: Any) -> TaskSet:
    """Multi-SWE-RL TaskSet."""
    from verifiers.envs.experimental.composable.tasksets.swe.multi_swe import (
        MultiSWETaskSet,
    )

    return MultiSWETaskSet(**kwargs)


def make_openswe_taskset(**kwargs: Any) -> TaskSet:
    """OpenSWE TaskSet."""
    from verifiers.envs.experimental.composable.tasksets.swe.openswe import (
        OpenSWETaskSet,
    )

    return OpenSWETaskSet(**kwargs)


def make_swelego_real_taskset(**kwargs: Any) -> TaskSet:
    """SWE-Lego Real-Data TaskSet (~4.4k resolved real GitHub issues, public images).

    Defaults to PrimeIntellect/SWE-Lego-Real-Data, a filtered fork of the
    upstream SWE-Lego/SWE-Lego-Real-Data that drops rows with truncated pytest
    parametrize test IDs in FAIL_TO_PASS / PASS_TO_PASS (11.5% of upstream).
    """
    from verifiers.envs.experimental.composable.tasksets.swe.swe_lego import (
        SWELegoTaskSet,
    )

    return SWELegoTaskSet(**kwargs)
