"""Config schemas for the v1 entrypoints — the objects the CLIs parse."""

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.configs.serve import EnvServerConfig

__all__ = ["EvalConfig", "EnvServerConfig"]
