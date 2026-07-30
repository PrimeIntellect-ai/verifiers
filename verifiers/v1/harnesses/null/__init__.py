"""Compatibility alias for historical configs; new SKX runs use ``id = "skx"``."""

from verifiers.v1.harnesses.skx import SkxHarness, SkxHarnessConfig

NullHarness = SkxHarness
NullHarnessConfig = SkxHarnessConfig

__all__ = ["NullHarness", "NullHarnessConfig"]
