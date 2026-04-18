# Verifiers v0.1.12.dev4 Release Notes

*Date:* 04/13/2026

## Highlights since v0.1.12.dev3

- Fixed CliAgentEnv getting permanently stuck when a Prime tunnel dies server-side.
- Simplified RLM message transcript handling.

## Changes included in v0.1.12.dev4 (since v0.1.12.dev3)

### Features and enhancements

- feat: detect server-side tunnel death and auto-recreate in CliAgentEnv (#1127)
- refactor: simplify RLM message transcript handling (#1116)

### Fixes and maintenance

- fix: raise AgentError when agent crashes before any LLM call (exit_code!=0 with empty trajectory) (#1127)
- fix: bump `prime-tunnel>=0.1.6` for `check_registered()` support (#1127)
- docs: prefix prime eval models (#1125)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.12.dev3...v0.1.12.dev4
