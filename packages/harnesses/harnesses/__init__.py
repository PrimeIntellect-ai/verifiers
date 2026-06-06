__version__ = "0.1.2"

from .command import CommandHarness, CommandHarnessConfig
from .mini_swe_agent import MiniSWEAgent, MiniSWEAgentConfig
from .opencode import OpenCode, OpenCodeConfig
from .pi import Pi, PiConfig
from .replay import ReplayHarness
from .rlm import RLM, RLMConfig
from .terminus_2 import Terminus2, Terminus2Config

LAZY_EXPORTS = {
    "NeMoGymHarness": (".nemo_gym", "NeMoGymHarness"),
    "NeMoGymHarnessConfig": (".nemo_gym", "NeMoGymHarnessConfig"),
}

__all__ = [
    "CommandHarness",
    "CommandHarnessConfig",
    "MiniSWEAgent",
    "MiniSWEAgentConfig",
    *LAZY_EXPORTS,
    "OpenCode",
    "OpenCodeConfig",
    "Pi",
    "PiConfig",
    "ReplayHarness",
    "RLM",
    "RLMConfig",
    "Terminus2",
    "Terminus2Config",
]


def __getattr__(name: str):
    if name in LAZY_EXPORTS:
        module_name, symbol_name = LAZY_EXPORTS[name]
        from importlib import import_module

        try:
            return getattr(import_module(module_name, __name__), symbol_name)
        except ModuleNotFoundError as exc:
            if exc.name in {"aiohttp", "nemo_gym", "omegaconf"}:
                raise ImportError(
                    f"To use {name}, install as `verifiers[nemogym]`."
                ) from exc
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
