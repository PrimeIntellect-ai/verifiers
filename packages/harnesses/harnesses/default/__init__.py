"""Default v1 harness and the program script it stages into the runtime."""

from harnesses.default.harness import DefaultHarness, DefaultHarnessConfig

Config = DefaultHarnessConfig
Harness = DefaultHarness

__all__ = ["Config", "Harness", "DefaultHarness", "DefaultHarnessConfig"]
