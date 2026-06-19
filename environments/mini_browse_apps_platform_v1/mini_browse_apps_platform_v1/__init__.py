"""mini-browse-apps-platform-v1 — sandboxed local-app Mini Browse browser tasks (v1).

Co-packages the taskset and its browser harness; both are resolved by id from this module's
`__all__` (`--taskset.id` / `--harness.id mini-browse-apps-platform-v1`).
"""

from .harness import MiniBrowseHarness, MiniBrowseHarnessConfig
from .taskset import MiniBrowseAppsConfig, MiniBrowseAppsTaskset, MiniBrowseAppTask

__all__ = [
    "MiniBrowseAppsTaskset",
    "MiniBrowseAppsConfig",
    "MiniBrowseAppTask",
    "MiniBrowseHarness",
    "MiniBrowseHarnessConfig",
]
