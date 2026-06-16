"""compact — an example custom harness that rewrites its context every turn."""

from compact.harness import CompactingHarness, CompactingHarnessConfig

Config = CompactingHarnessConfig
Harness = CompactingHarness

__all__ = ["Config", "Harness", "CompactingHarness", "CompactingHarnessConfig"]
