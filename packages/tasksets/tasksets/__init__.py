import importlib

from .harbor import HarborTaskset, HarborTasksetConfig

__all__ = [
    "HarborTaskset",
    "HarborTasksetConfig",
    "OpenEnvTaskset",
    "OpenEnvTasksetConfig",
    "OpenRewardTaskset",
    "OpenRewardTasksetConfig",
    "TextArenaTaskset",
    "TextArenaTasksetConfig",
]


def __getattr__(name: str):
    if name in ("OpenEnvTaskset", "OpenEnvTasksetConfig"):
        module = importlib.import_module("tasksets.openenv")
        return getattr(module, name)
    if name in ("OpenRewardTaskset", "OpenRewardTasksetConfig"):
        module = importlib.import_module("tasksets.openreward")
        return getattr(module, name)
    if name in ("TextArenaTaskset", "TextArenaTasksetConfig"):
        module = importlib.import_module("tasksets.textarena")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
