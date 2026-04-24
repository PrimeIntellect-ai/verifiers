# Verifiers v0.1.13.dev1 Release Notes

*Date:* 04/18/2026

## Highlights since v0.1.12

- Added the SWELego-Real TaskSet (PrimeIntellect fork, filtered upstream) for broader SWE benchmark coverage.
- Added `timeout_minutes` kwarg to R2E, SWEBench, Multi-SWE, and OpenSWE tasksets for finer-grained per-task timeout control.
- Surfaced agent timeout as `state['error']` in `CliAgentEnv` so timeouts are visible in eval results.
- Fixed `CliAgentEnv` poll loop to honor `self.poll_interval` consistently.
- Bumped `prime-sandboxes` to 0.2.20.

## Changes included in v0.1.13.dev1 (since v0.1.12)

### Features and enhancements

- feat: add SWELego-Real TaskSet (PrimeIntellect fork, filtered upstream) (#1149)
- feat: add `timeout_minutes` kwarg to R2E / SWEBench / Multi-SWE / OpenSWE tasksets (#1171)

### Fixes and maintenance

- fix: surface agent timeout as `state['error']` in `CliAgentEnv` (#1170)
- fix: honor `self.poll_interval` in `CliAgentEnv` poll loop (#1173)
- chore: bump prime-sandboxes to 0.2.20 (#1174)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.12...v0.1.13.dev1
