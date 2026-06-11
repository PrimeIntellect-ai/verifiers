"""Config schemas for the v1 entrypoints — the objects the CLIs parse."""

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.configs.serve import ServeConfig

__all__ = ["EvalConfig", "ServeConfig"]
