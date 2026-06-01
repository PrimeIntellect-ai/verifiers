__version__ = "0.1.4"

from .harbor import HarborTaskset, HarborTasksetConfig

LAZY_EXPORTS = {
    "NeMoGymTaskset": (".nemo_gym", "NeMoGymTaskset"),
    "NeMoGymTasksetConfig": (".nemo_gym", "NeMoGymTasksetConfig"),
    "OpenEnvTaskset": (".openenv", "OpenEnvTaskset"),
    "OpenEnvTasksetConfig": (".openenv", "OpenEnvTasksetConfig"),
    "OpenRewardTaskset": (".openreward", "OpenRewardTaskset"),
    "OpenRewardTasksetConfig": (".openreward", "OpenRewardTasksetConfig"),
    "TextArenaTaskset": (".textarena", "TextArenaTaskset"),
    "TextArenaTasksetConfig": (".textarena", "TextArenaTasksetConfig"),
}

__all__ = [
    "HarborTaskset",
    "HarborTasksetConfig",
    *LAZY_EXPORTS,
]


def __getattr__(name: str):
    if name in LAZY_EXPORTS:
        module_name, symbol_name = LAZY_EXPORTS[name]
        from importlib import import_module

        return vars(import_module(module_name, __name__))[symbol_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
