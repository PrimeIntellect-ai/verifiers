"""Default endpoint-backed tool-loop harness as a swappable package.

Exposes `BaseHarnessConfig` and `load_harness(config)` so the entrypoint
can address the default harness the same way it addresses more advanced harnesses.
"""

from __future__ import annotations

from ...config import HarnessConfig
from ...harness import Harness


class BaseHarnessConfig(HarnessConfig):
    """Config for the default endpoint-backed harness (`vf.Harness`)."""


def load_harness(config: BaseHarnessConfig | None = None) -> Harness:
    return Harness(config=config)
