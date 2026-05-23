import importlib

from .harbor import HarborTaskset, HarborTasksetConfig

__all__ = [
    "HarborTaskset",
    "HarborTasksetConfig",
    "TextArenaTaskset",
    "TextArenaTasksetConfig",
]


def __getattr__(name: str):
    if name in ("TextArenaTaskset", "TextArenaTasksetConfig"):
        module = importlib.import_module("verifiers.v1.packages.tasksets.textarena")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
