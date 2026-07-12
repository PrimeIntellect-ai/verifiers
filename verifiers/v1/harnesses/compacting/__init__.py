"""compacting — the default harness plus rlm-style context compaction: once the next
prompt's reported or estimated token count crosses `compact_at_tokens`, the chat loop asks
the model for a non-empty handoff summary (with the full conversation still in context) and
rebuilds its messages as `[system, user(framing + summary)]`, branching the trace at the
rewrite. Resolved by id via
`load_harness` (`--harness.id compacting`).

(The id is `compacting`, not `compact` — `compact` is taken by the example context-rewrite
harness under `environments/compact`.)"""

from verifiers.v1.harnesses.compacting.harness import (
    CompactingHarness,
    CompactingHarnessConfig,
)

__all__ = ["CompactingHarness", "CompactingHarnessConfig"]
