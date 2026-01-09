# Verifiers v0.1.2.post0 – Release Notes

Minor post-release update focusing on polish: CLI script bug fixes and enhancements, environment example cleanup, better reporting, and improved test coverage.

## Highlights
- vf-eval: fixed rollout indexing bugs and improved reliability when sampling multiple rollouts.
- vf-init: streamlined project initialization and naming (removed automatic `vf-` prefix) and refreshed templates.
- Environments: documentation and prompt cleanups; added/updated AIME examples; improved report embedding.
- Tests: expanded coverage across rubric behavior, XML parser, and environment edge cases.

## Changes by Area
### CLI and Scripts
- vf-eval
  - Fix index handling when using multiple rollouts (PR #197).
  - Ensure metrics columns are included in generated datasets via supporting utilities (PR #194).
- vf-init
  - Remove automatic `vf-` prefix during init to honor provided names (PR #190).
  - Update README template/content for new environments (multiple small tweaks).

### Environments and Examples
- AIME 2024 / AIME 2025 updates (PR #199).
- Math Python example: prompt/readme/report cleanups.
- General environment cleanup and README refreshes across multiple examples.
- HotpotQA example: troubleshooting notes and minor fixes.

### Parsers, Rubrics, and Utils
- XMLParser: fix handling of string completions during `parse_answer` (PR #196).
- Rubric: ensure error-handling behavior is well-covered by tests (PR #195).
- Reporting: improvements to report generation/embedding (`report_utils`).
- Dataset helpers: include metrics columns in outputs where expected (PR #194).

### Tests
- Increase test coverage for:
  - Rubric error handling (PR #195).
  - XML parser behavior (new tests).
  - Environment edge cases and extra scenarios.

## Acknowledgements
Thank you to everyone who contributed to this minor release:
- [@snellingio](https://github.com/snellingio)
- [@vgel](https://github.com/vgel)
- [@JannikSt](https://github.com/JannikSt)
- [@samsja](https://github.com/samsja)

If we missed anyone, thank you as well—your contributions are appreciated.

## Upgrade Notes
- No breaking API changes.
- When initializing a new environment with `vf-init`, note the name is now used verbatim (no automatic `vf-` prefix, PR #190).

## Reference Commits (since v0.1.2)
- Fix XMLParser string completion parsing (PR #196)
- Improve test coverage for Rubric error handling (PR #195)
- Include metrics columns in dataset outputs (PR #194)
- Fix vf-eval rollout index handling (PR #197)
- Remove automatic `vf-` prefix from init (PR #190)
- AIME 2024 / 2025 environments updates (PR #199)
- Environment README/reporting cleanups and misc improvements

## Full Changelog
- [v0.1.2...v0.1.2.post0](https://github.com/willccbb/verifiers/compare/v0.1.2...v0.1.2.post0)
