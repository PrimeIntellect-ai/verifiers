__version__ = "0.1.3"

from .mini_swe_agent import MiniSWEAgent, MiniSWEAgentConfig, MiniSWEAgentProgramConfig
from .opencode import OpenCode, OpenCodeConfig, OpenCodeProgramConfig
from .pi import Pi, PiConfig, PiProgramConfig
from .replay import ReplayHarness
from .rlm import RLM, RLMConfig, RLMProgramConfig
from .terminus_2 import Terminus2, Terminus2Config, Terminus2ProgramConfig

LAZY_EXPORTS = {
    "NeMoGymHarness": (".nemo_gym", "NeMoGymHarness"),
    "NeMoGymHarnessConfig": (".nemo_gym", "NeMoGymHarnessConfig"),
}

__all__ = [
    "MiniSWEAgent",
    "MiniSWEAgentConfig",
    "MiniSWEAgentProgramConfig",
    *LAZY_EXPORTS,
    "OpenCode",
    "OpenCodeConfig",
    "OpenCodeProgramConfig",
    "Pi",
    "PiConfig",
    "PiProgramConfig",
    "ReplayHarness",
    "RLM",
    "RLMConfig",
    "RLMProgramConfig",
    "Terminus2",
    "Terminus2Config",
    "Terminus2ProgramConfig",
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
