# Verifiers v0.1.13.dev4 Release Notes

*Date:* 04/22/2026

## Highlights since v0.1.13.dev3

- RLM harness: new `rlm_tools` kwarg sets both `Harness.tool_names` (for `ToolMonitorRubric`) and the sandbox `RLM_TOOLS` env var from a single source, plus new `Harness.environment_vars` field merged harness-wins-on-conflict by `ComposableEnv`.
- Refactored experimental RLM checkout caching; `DEFAULT_RLM_BRANCH` renamed to `DEFAULT_RLM_REF` and `rlm_harness(..., rlm_branch=...)` renamed to `rlm_ref=` to reflect that any git ref (branch, tag, sha) is accepted.
- Added `SandboxTimeouts` dataclass centralizing per-operation sandbox HTTP timeouts.
- Expanded task coverage with SWE-rebench-V2 and a multilingual SWESmith taskset, plus a `filter_fn` kwarg on all tasksets for ad-hoc row filtering.
- `vf-eval`: renamed `-d/--debug` to `--disable-tui` and `--tui` to `--fullscreen` for clearer intent.
- RLM rollout metrics (context tokens, programmatic tool calls) exposed to verifiers and auto-merged by the composable env.

## Changes included in v0.1.13.dev4 (since v0.1.13.dev3)

### Features and enhancements

- vf-eval: replace -d/--debug with --disable-tui, rename --tui to --fullscreen (#1183)
- Expose RLM metrics to verifiers (#1195)
- Add streaming observability + resume to TaskSet.validate() (#1169)
- Refactor experimental RLM checkout caching (#1202)
- feat: add filter_fn kwarg to all tasksets for ad-hoc row filtering (#1199)
- feat: add multilingual SWESmithTaskSet (#1186)
- feat: add SWE-rebench-V2 TaskSet (#1187)
- HarborMCPMixin (#1146)
- feat: SandboxTimeouts dataclass — centralize per-operation sandbox HTTP timeouts (#1207)
- Run SWE-Lego eval via dataset's canonical test_cmd (#1205)
- Authenticate interception server via INTERCEPTION_SECRET (#1180)
- feat: revert agent test edits at grading (swe_lego, swe_rebench_v2) (#1212)
- AgentError: rollout_id, sandbox_id, ... (#1218)
- Remove RLM_DEFAULT_TOOL_NAMES, accept rlm_tools (#1223)
- r2e_gym: add hide_tests_from_agent flag + expose instance_id/repo aliases (#1208)
- feat(rlm): upload a /usr/local/bin/git shim, gated by ``allow_git`` (#1225)

### Fixes and maintenance

- Keep harness metrics merge inside experimental composable env (#1201)
- Propagate typed exceptions from SWE/Harbor validate_instance (#1204)
- fix: pass explicit 60s timeout to get_background_job in poll_job_completion (#1206)
- fix: bump opencode harness default release to v1.1.63-rl2 (#1184)
- validate(): extract resume-file parsing into a named helper (#1209)
- fix: SandboxTimeouts fields must be int (sidecar deserializes as u64) (#1210)
- fix: respect framework-injected OPENAI_API_KEY in RLM and opencode harnesses (#1213)
- fix: offload composable _upload_dir tar build to thread (#1224)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.13.dev3...v0.1.13.dev4
