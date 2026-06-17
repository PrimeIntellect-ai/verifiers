"""compact — an example custom harness: rewrites its context every turn (so the
trajectory branches, one branch per turn). Resolved by id via `load_harness`."""

from compact.harness import CompactingHarness, CompactingHarnessConfig

__all__ = ["CompactingHarness", "CompactingHarnessConfig"]
