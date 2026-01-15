"""
Browser Environment - Unified browser automation with DOM and CUA modes.

Usage:
    from verifiers.envs.integrations.browser_env import load_environment

    # DOM mode (natural language)
    env = load_environment(mode="dom", benchmark="gaia")

    # CUA mode (vision-based)
    env = load_environment(mode="cua", benchmark="webvoyager")

Install:
    uv add 'verifiers[browser]'
"""

import importlib
from typing import TYPE_CHECKING

# Always available - no special dependencies
from .browser_datasets import (
    load_benchmark_dataset,
    load_smoke_test_dataset,
    load_gaia_dataset,
    load_webvoyager_dataset,
    load_mind2web_dataset,
    BenchmarkType,
)
from .rewards import (
    efficiency_reward,
    judge_answer_reward,
    get_judge_prompt,
    JUDGE_PROMPT,
    TASK_JUDGE_PROMPT,
)

# Lazy imports for classes that require optional dependencies (stagehand)
_LAZY_IMPORTS = {
    "BrowserEnv": ".browser_env:BrowserEnv",
    "ModeType": ".browser_env:ModeType",
    "load_environment": "._loader:load_environment",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name].rsplit(":", 1)
        try:
            module = importlib.import_module(module_path, package=__name__)
            return getattr(module, attr_name)
        except ImportError as e:
            if "stagehand" in str(e):
                raise ImportError(
                    f"To use {name}, install the browser dependencies: "
                    "uv add 'verifiers[browser]' or pip install 'verifiers[browser]'"
                ) from e
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main exports (lazy)
    "BrowserEnv",
    "ModeType",
    "load_environment",
    # Dataset loading (always available)
    "BenchmarkType",
    "load_benchmark_dataset",
    "load_smoke_test_dataset",
    "load_gaia_dataset",
    "load_webvoyager_dataset",
    "load_mind2web_dataset",
    # Rewards (always available)
    "efficiency_reward",
    "judge_answer_reward",
    "get_judge_prompt",
    "JUDGE_PROMPT",
    "TASK_JUDGE_PROMPT",
]

if TYPE_CHECKING:
    from .browser_env import BrowserEnv, ModeType
    from ._loader import load_environment
