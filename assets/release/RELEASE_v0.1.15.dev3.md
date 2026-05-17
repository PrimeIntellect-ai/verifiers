# Verifiers v0.1.15.dev3 Release Notes

*Date:* 05/13/2026

## Highlights since v0.1.15.dev2

- **OverlongPromptError no longer surfaces as a spurious interception error.** `CliAgentEnv` / `ComposableEnv` rollouts that overflow the model context now report a clean `stop=prompt_too_long` without an accompanying `InterceptionError` in the eval `errors` summary. The interception server's first-error-wins guard now also short-circuits once the rollout loop has finalized via `state["prompt_too_long"]`, so tail-end failures (e.g. `write_eof` to an agent that already closed its transport) can no longer re-attach a spurious `StreamInterrupted` on top of the real stop signal.
- **v1 sandbox commands run as background jobs.** v1 sandbox program commands and Harbor verifier tests now run through `run_background_job` instead of foreground `execute_command`, preserving configured command/test timeouts as the background-job budget.

## Changes included in v0.1.15.dev3 (since v0.1.15.dev2)

### Fixes and maintenance

- fix: treat OverlongPromptError as stop condition in interception proxy (#1365)
- Use background jobs for v1 sandbox commands (#1364)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev2...v0.1.15.dev3
