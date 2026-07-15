"""direct — v1's built-in in-process harness: a chat loop with no subprocess and no tools.
The cheapest episode (e.g. an in-process judge agent in a topology). Resolved by id via
`load_harness`."""

from verifiers.v1.harnesses.direct.harness import DirectHarness, DirectHarnessConfig

__all__ = ["DirectHarness", "DirectHarnessConfig"]
