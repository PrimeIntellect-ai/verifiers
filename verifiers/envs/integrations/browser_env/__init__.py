"""
Browser Environment - Unified browser automation with DOM and CUA modes.

Usage:
    from verifiers.envs.integrations.browser_env import BrowserEnv

    # DOM mode - natural language browser automation
    env = BrowserEnv(
        mode="dom",
        dataset=my_dataset,
        rubric=my_rubric,
    )

    # CUA mode with sandbox (default) - automatic server deployment
    env = BrowserEnv(
        mode="cua",
        dataset=my_dataset,
        rubric=my_rubric,
    )

    # CUA mode without sandbox - requires manual server setup
    env = BrowserEnv(
        mode="cua",
        use_sandbox=False,
        server_url="http://localhost:3000",
        dataset=my_dataset,
        rubric=my_rubric,
    )

Install:
    uv add 'verifiers[browser]'
"""

import importlib
from typing import TYPE_CHECKING

# Lazy imports for classes that require optional dependencies (stagehand)
_LAZY_IMPORTS = {
    "BrowserEnv": ".browser_env:BrowserEnv",
    "ModeType": ".browser_env:ModeType",
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
    "BrowserEnv",
    "ModeType",
]

if TYPE_CHECKING:
    from .browser_env import (
        BrowserEnv,
        ModeType,
    )
