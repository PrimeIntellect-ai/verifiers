# Verifiers v0.1.13.dev6 Release Notes

*Date:* 04/23/2026

## Highlights since v0.1.13.dev5

- `rlm_harness` is now the single source of truth for RLM_* sandbox env vars. New kwargs `rlm_max_turns`, `rlm_max_turns_in_context`, `rlm_exec_timeout` map 1:1 onto the matching env vars on `Harness.environment_vars` and merge into the sandbox via `ComposableEnv.build_env_vars` (harness-wins). Research envs can stop setting these via `ComposableEnv(environment_vars=…)` — pass them through as harness kwargs instead.
- `TaskSet.filter` / `.take` now return `Self`, not `TaskSet`, so subclass types survive taskset chaining for downstream typed consumers.

## Changes included in v0.1.13.dev6 (since v0.1.13.dev5)

### Features and enhancements

- rlm_harness: own RLM_MAX_TURNS / _IN_CONTEXT / _EXEC_TIMEOUT env vars (#1229)

### Fixes and maintenance

- types: TaskSet.filter / .take return Self, not TaskSet (#1232)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.13.dev5...v0.1.13.dev6
