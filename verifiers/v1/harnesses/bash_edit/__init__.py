"""bash_edit — v1's built-in agentic harness: the `bash` chat loop plus a local `edit` tool
(single-occurrence string replacement, ported from rlm). Its `harness.py` (class + config) and
the `program.py` script staged into the runtime. Resolved by id via `load_harness`."""

from verifiers.v1.harnesses.bash_edit.harness import (
    BashEditHarness,
    BashEditHarnessConfig,
)

__all__ = ["BashEditHarness", "BashEditHarnessConfig"]
