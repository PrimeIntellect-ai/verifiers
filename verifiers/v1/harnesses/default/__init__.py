"""default — v1's built-in agentic harness (the fallback when no `--harness.id` is given): a chat
loop with a local `bash` tool and an optional `edit` tool (single-occurrence string replacement,
on by default). Its `harness.py` (class + config) and the `program.py` script staged into the
runtime. Resolved by id via `load_harness`."""

from verifiers.v1.harnesses.default.harness import (
    DefaultHarness,
    DefaultHarnessConfig,
)

__all__ = ["DefaultHarness", "DefaultHarnessConfig"]
