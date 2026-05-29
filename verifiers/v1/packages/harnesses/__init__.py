import importlib
from typing import TYPE_CHECKING

from .configs import (
    MiniSWEAgentConfig,
    OpenCodeConfig,
    PiConfig,
    RLMConfig,
    Terminus2Config,
)
from .mini_swe_agent import MiniSWEAgent
from .opencode import OpenCode
from .pi import Pi
from .rlm import RLM
from .terminus_2 import Terminus2

__all__ = [
    "MiniSWEAgent",
    "NeMoGymHarness",
    "NeMoGymHarnessConfig",
    "MiniSWEAgentConfig",
    "OpenCode",
    "OpenCodeConfig",
    "Pi",
    "PiConfig",
    "RLM",
    "RLMConfig",
    "Terminus2",
    "Terminus2Config",
]

_LAZY_IMPORTS = {
    "NeMoGymHarness": "verifiers.v1.packages.harnesses.nemo_gym:NeMoGymHarness",
    "NeMoGymHarnessConfig": (
        "verifiers.v1.packages.harnesses.nemo_gym:NeMoGymHarnessConfig"
    ),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name].split(":")
        try:
            return getattr(importlib.import_module(module_path), attr)
        except ModuleNotFoundError as exc:
            if exc.name == "aiohttp":
                raise ImportError(
                    f"To use {name}, install as `verifiers[nemogym]`."
                ) from exc
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .nemo_gym import NeMoGymHarness, NeMoGymHarnessConfig
