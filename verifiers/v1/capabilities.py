"""Shared capability names for v1 harness/taskset compatibility checks."""

from enum import StrEnum


class HarnessCapability(StrEnum):
    """Coarse-grained contracts a taskset may require from its selected harness."""

    BROWSER_CONTROL = "browser-control"
    """Runs browser-control tasks from an agreed runtime payload contract."""
