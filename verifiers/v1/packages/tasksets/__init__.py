import importlib

from .harbor import HarborTaskset, HarborTasksetConfig
from .nemo_gym import NeMoGymTaskset, NeMoGymTasksetConfig

__all__ = [
    "HarborTaskset",
    "HarborTasksetConfig",
    "NeMoGymTaskset",
    "NeMoGymTasksetConfig",
    "TextArenaTaskset",
    "TextArenaTasksetConfig",
]


def __getattr__(name: str):
    if name in ("TextArenaTaskset", "TextArenaTasksetConfig"):
        module = importlib.import_module("verifiers.v1.packages.tasksets.textarena")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
