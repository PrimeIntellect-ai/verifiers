"""bash — v1's built-in agentic harness: the `default` chat loop plus a single local `bash`
tool. Its `harness.py` (class + config) and the `program.py` script staged into the runtime.
Resolved by id via `load_harness`."""

from harnesses.bash.harness import BashHarness, BashHarnessConfig

__all__ = ["BashHarness", "BashHarnessConfig"]
