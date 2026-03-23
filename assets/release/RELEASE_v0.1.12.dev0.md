# Verifiers v0.1.12.dev0 Release Notes

*Date:* 03/22/2026

## Highlights since v0.1.11

- Added the new `opencode_rlm_env` plus broader opencode environment cleanups and performance work, including executor autoscaling and incremental metrics for better throughput under load.
- Improved runtime reliability with more robust cancellation handling, env server startup tuning, safer port allocation, and better event loop lag monitoring.
- Expanded configuration and lifecycle ergonomics with `Rubric` cleanup/teardown hooks, configurable eval `output_dir`, preserved multimodal media in saved eval results, and `ClientConfig.extra_headers_from_state`.
- Refined rubric and environment performance with lazy imports, faster file I/O, math-rubric optimizations, and follow-up fixes to logging and response normalization.
- Added fresh docs for BrowserEnv integration and a new environment performance guide.

## Changes included in v0.1.12.dev0 (since v0.1.11)

### Environments, execution, and performance

- deprecate `RolloutGatewayMixin` (#1017)
- Lazily import packages (#1019)
- Add BrowserEnv integration README (#1020)
- tune GC on env server before accepting requests (#1022)
- opencode_rlm_env (#1023)
- Preserve multimodal media in saved eval results (#1015)
- Add cleanup and teardown lifecycle hooks to Rubric (#1026)
- remove redundant msg normalization + align `env_response` api (#1027)
- make `output_dir` configurable in evals (#1029)
- misc improvements to opencode envs (#999)
- perf improvs for opencode envs + math rubric (#1034)
- fix: task cancelation race + RLM sandbox workers (#1035)
- perf: incremental metrics (#1036)
- perf: offload file i/o to thread pool (#1037)
- feat: improve event loop lag monitor (#1038)
- perf: executor autoscaling (#1039)

### Reliability, cancellation, and bug fixes

- Fix get_free_port_pair() TOCTOU race condition (#1013)
- fix display of custom sampling args (#1025)
- fix output dir logging (#1041)
- fix: revert opencode_env config regression and move RLM logic out of cli_agent_env (#1042)
- chore: reuse math rubric in hybrid math rubric (#1043)
- fix cancelled + serialize error (#1044)
- docs: add performance guide for environments (#1045)
- perf: math rubric skip overlong answers (#1046)
- fix: call uncancel() after catching `CancelledError` in `process_request` (#1047)
- feat: add `extra_headers_from_state` to `ClientConfig` (#1048)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.11...v0.1.12.dev0
