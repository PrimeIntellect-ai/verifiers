# Verifiers v0.1.15.dev5 Release Notes

*Date:* 05/14/2026

## Highlights since v0.1.15.dev4

- **Terminus2 is now a bundled v1 CLI harness.** `vf.Terminus2` runs Harbor's Terminus 2 agent behind the v1 harness boundary, is exported from `verifiers.v1` and the root `verifiers` namespace, uses OpenAI-compatible proxy defaults, and captures optional Terminus2 logs as artifacts.
- **Harbor sandbox defaults stay explicit.** Harbor tasksets and task freezing now preserve only task-provided sandbox settings, including `network_access` only when task config explicitly sets internet access.

## Changes included in v0.1.15.dev5 (since v0.1.15.dev4)

### v1 harnesses and tasksets

- Add Terminus2 harness (#1356)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev4...v0.1.15.dev5
