"""default — v1's built-in harness: its `harness.py` (class + config) and the
`program.py` script it stages into the runtime. Resolved by id via `load_harness`."""

from default.harness import DefaultHarness, DefaultHarnessConfig


def load_harness(config: DefaultHarnessConfig) -> DefaultHarness:
    return DefaultHarness(config)


__all__ = ["DefaultHarness", "DefaultHarnessConfig", "load_harness"]
