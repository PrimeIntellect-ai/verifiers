# Verifiers v0.1.13.dev5 Release Notes

*Date:* 04/22/2026

## Highlights since v0.1.13.dev4

- Made the interception proxy's streaming response resilient to upstream cuts: 10s SSE keepalive comments keep idle streams warm, per-chunk `asyncio.sleep(0)` forces an event-loop yield so content and close can't race the transport flush under warmup-burst contention, and transport exceptions at prepare/write/write_eof are surfaced as `StreamInterrupted` into `state["error"]` so rollouts reschedule instead of looking like clean zero-turn completions.
- Added a new experimental `mini_swe_agent` composable harness (pip/uv install with SHA256-verified wheel download), exported alongside existing `rlm` and `opencode` harnesses.
- Extended `SandboxMixin` to cover VM sandboxes in addition to containers (including GPU VMs via `CreateSandboxRequest`), with documentation clarifying feature parity (file I/O, background jobs, cleanup) and container-only features (port exposure, SSH).

## Changes included in v0.1.13.dev5 (since v0.1.13.dev4)

### Latest changes from main

- Includes the latest `main` changes through the interception proxy streaming resilience fix (#1194), along with the `mini_swe_agent` harness (#1219) and `SandboxMixin` VM sandbox support/docs (#1222).

### Features and enhancements

- Add mini-swe-agent harness (#1219)
- Update `SandboxMixin` (#1222)

### Fixes and maintenance

- fix: make interception proxy streaming resilient to upstream cuts (#1194)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.13.dev4...v0.1.13.dev5
