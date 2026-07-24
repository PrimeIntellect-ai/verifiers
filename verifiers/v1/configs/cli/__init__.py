"""The CLI entrypoints' run configs — each `uv run <cmd>`'s parsed tree."""

from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.cli.debug import DebugConfig
from verifiers.v1.configs.cli.init import InitConfig
from verifiers.v1.configs.cli.serve import ServeConfig
from verifiers.v1.configs.cli.validate import ValidateConfig

__all__ = ["DebugConfig", "EvalConfig", "InitConfig", "ServeConfig", "ValidateConfig"]
