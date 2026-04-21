# Verifiers v0.1.13.dev3 Release Notes

*Date:* 04/19/2026

## Highlights since v0.1.13.dev2

- Propagated interception-stream write failures into rollout state as `StreamInterrupted` so truncated agent streams no longer surface as silent clean exits.
- Made RLM checkout resolution lazy in the composable harness, so loading RLM-based environments no longer clones the private checkout up front.

## Changes included in v0.1.13.dev3 (since v0.1.13.dev2)

### Features and enhancements

- make download of rlm lazy (#1192)

### Fixes and maintenance

- fix: propagate interception-stream cuts into rollout state (#1191)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.13.dev2...v0.1.13.dev3
