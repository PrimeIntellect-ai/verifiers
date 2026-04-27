# Verifiers v0.1.13.dev7 Release Notes

*Date:* 04/24/2026

## Highlights since v0.1.13.dev6

- `rlm_harness` swaps turn-based context caps for token-based auto-compaction: new `summarize_at_tokens: int | None` kwarg maps to `RLM_SUMMARIZE_AT_TOKENS`, while `rlm_max_turns_in_context` / `RLM_MAX_TURNS_IN_CONTEXT` are removed to match upstream `rlm`. `summarize` also drops out of the default `rlm_tools` set. Invalid shapes fail at harness-build time instead of deep inside the sandbox.
- Reverted `TaskSet.filter` / `.take` returning `Self` (originally #1232) — the change broke Python 3.10/3.11 compatibility. CI now exercises the 3.10 and 3.11 test matrices so the fix can be restored with confidence.

## Changes included in v0.1.13.dev7 (since v0.1.13.dev6)

### Features and enhancements

- rlm_harness: add `summarize_at_tokens`, drop `rlm_max_turns_in_context` (#1236)

### Fixes and maintenance

- Revert "types: TaskSet.filter / .take return Self, not TaskSet (#1232)" (#1237)
- ci: add Python 3.10 and 3.11 to the test matrix (#1237)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.13.dev6...v0.1.13.dev7
