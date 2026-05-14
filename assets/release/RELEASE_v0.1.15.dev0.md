# Verifiers v0.1.15.dev0 Release Notes

*Date:* 05/12/2026

## Highlights since v0.1.14

- **v1 taskset/harness follow-through.** Refreshes the docs and examples around the v1 taskset/harness path, completes the `opencode_harbor` migration, adds setup-state compatibility for environments, and makes sandbox worker caps configurable.
- **Agent and SWE environment updates.** Adds the experimental SWE debug environment, moves OpenCode configuration into reusable harness code, refactors the LangChain Deep Agents example to a v1 Wikispeedia taskset, and removes the direct `prime-sandboxes` dependency from `rlm_swe_v1`.
- **Interception, sandbox, and renderer hardening.** Requires auth for interception endpoints, surfaces swallowed interception response errors, fixes keepalive configuration, avoids holding sandbox workers while background jobs run, retries transient file-read timeouts, and aligns renderer tool envelopes with the training distribution.

## Changes included in v0.1.15.dev0 (since v0.1.14)

### Features and enhancements

- feat(composable): render pre-compaction branches via harness hook (#1291)
- Make sandbox worker caps configurable (#1305)
- Support `setup_state` return-state compatibility (#1308)
- Update v1 Taskset/Harness docs and `opencode_harbor` migration (#1309)
- Refactor deep-agents envs to v1 taskset-harness and add Wikispeedia example (#1317)
- Add SWE debug environment (#1306)
- Move OpenCode config into reusable harness package (#1318)
- Remove direct `prime-sandboxes` dependency from `rlm_swe_v1` (#1316)
- Install sandbox fn program packages (#1319)
- opencode: write config under `XDG_CONFIG_HOME` (#1320)
- feat: surface swallowed `response_future` errors in interception proxy (#1196)
- opencode: pin `small_model` to the intercepted provider (#1323)
- Avoid holding sandbox workers with `run_background_job()` (#1328)
- Skip OpenCode install when binary is pre-baked (#1176)

### Fixes and maintenance

- Require auth for interception endpoints (#1304)
- fix: mute httpcore/httpx DEBUG when env-worker `json_logging` sets root level (#1234)
- bump interception client size to handle trajectories with images (#1310)
- Fix eval summary for heterogeneous metrics (#1295)
- apt: harden sandbox bootstrap against transient `archive.ubuntu.com` flakes (#1284)
- fix(renderer-client): wrap tools in OpenAI envelope to match training distribution (#1307)
- fixes rlm compaction prompt so we still get tito hit (#1322)
- Fix interception keepalive env override (#1321)
- fix: narrow `send_cancel` `BaseException` catch to `Exception` (#1198)
- Fix Harbor Hub task root handling (#1345)
- Pin OpenCode small model in legacy configs (#1327)
- Remove local training config scaffolding from Verifiers (#1348)
- retry Read file timed out (#1350)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.14...v0.1.15.dev0
