"""null — v1's built-in tool-less harness: its `harness.py` (class + config) and the
`program.py` script it stages into the runtime. A plain chat loop with the task's MCP tools
and no tools of its own. Resolved by id via `load_harness`."""

from verifiers.v1.harnesses.null.harness import NullHarness, NullHarnessConfig

__all__ = ["NullHarness", "NullHarnessConfig"]
