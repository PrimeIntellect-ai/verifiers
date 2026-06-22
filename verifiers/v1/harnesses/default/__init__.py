"""default — v1's built-in harness: its `harness.py` (class + config) and the
`program.py` script it stages into the runtime. Resolved by id via `load_harness`."""

from verifiers.v1.harnesses.default.harness import DefaultHarness, DefaultHarnessConfig

__all__ = ["DefaultHarness", "DefaultHarnessConfig"]
