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
        "swerebench-v2": make_swerebench_v2_taskset,
        "swesmith-py": make_swesmith_py_taskset,
        "swesmith-go": make_swesmith_go_taskset,
        "swesmith-java": make_swesmith_java_taskset,
        "swesmith-js": make_swesmith_js_taskset,
        "swesmith-ts": make_swesmith_ts_taskset,
        "swesmith-rs": make_swesmith_rs_taskset,
        "swesmith-cpp": make_swesmith_cpp_taskset,
        "swesmith-php": make_swesmith_php_taskset,
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


def make_swerebench_v2_taskset(**kwargs: Any) -> TaskSet:
    """SWE-rebench-V2 TaskSet (nebius/SWE-rebench-V2, 32k rows, 20 languages).

    Pass ``language=<one of the 20 language labels>`` to filter to a single
    language; omit for the full cross-language mix.
    """
    from verifiers.envs.experimental.composable.tasksets.swe.swe_rebench_v2 import (
        SWERebenchV2TaskSet,
    )

    return SWERebenchV2TaskSet(**kwargs)


def make_swesmith_taskset(language: str = "py", **kwargs: Any) -> TaskSet:
    """SWE-Smith TaskSet for a given language.

    SWE-Smith is a multilingual synthetic bug-fix dataset from the SWE-bench
    authors. One dataset / Docker image set per language
    (``SWE-bench/SWE-smith-{language}``). Scoring uses the upstream
    ``swesmith`` per-repo profile: runs the profile's ``test_cmd``, parses the
    output with the profile's language-specific ``log_parser``, and checks
    that every F2P and P2P test is marked PASSED.

    C++ currently has ~10% profile coverage (20/69 repos); rows without a
    profile are filtered out. All other priority languages have full coverage.

    Args:
        language: one of py, go, java, js, ts, rs, cpp, php.
        **kwargs: forwarded to ``SWESmithTaskSet``.
    """
    from verifiers.envs.experimental.composable.tasksets.swe.swe_smith import (
        SWESmithTaskSet,
    )

    return SWESmithTaskSet(language=language, **kwargs)


def make_swesmith_py_taskset(**kwargs: Any) -> TaskSet:
    return make_swesmith_taskset("py", **kwargs)


def make_swesmith_go_taskset(**kwargs: Any) -> TaskSet:
    return make_swesmith_taskset("go", **kwargs)


def make_swesmith_java_taskset(**kwargs: Any) -> TaskSet:
    return make_swesmith_taskset("java", **kwargs)


def make_swesmith_js_taskset(**kwargs: Any) -> TaskSet:
    return make_swesmith_taskset("js", **kwargs)


def make_swesmith_ts_taskset(**kwargs: Any) -> TaskSet:
    return make_swesmith_taskset("ts", **kwargs)


def make_swesmith_rs_taskset(**kwargs: Any) -> TaskSet:
    return make_swesmith_taskset("rs", **kwargs)


def make_swesmith_cpp_taskset(**kwargs: Any) -> TaskSet:
    return make_swesmith_taskset("cpp", **kwargs)


def make_swesmith_php_taskset(**kwargs: Any) -> TaskSet:
    """SWE-Smith PHP TaskSet — upstream dataset is a 1-row placeholder as of 2026-04."""
    return make_swesmith_taskset("php", **kwargs)
