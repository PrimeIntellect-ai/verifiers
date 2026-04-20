# Verifiers v0.1.12 Release Notes

*Date:* 04/17/2026

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.11...v0.1.12

## Highlights since v0.1.11

- Landed a new composable Task/Agent/Environment architecture and upstreamed opencode/RLM harnesses and swe/lean/math/cp/harbor tasksets into `verifiers.envs.experimental.composable`, so downstream environments can depend on them directly instead of via research-environments.
- Major `RLMEnv` overhaul: new `RLMPromptBuilder`, context dropping with `summarize_turns`, `max_turns_in_context`, sub-LLM toggle, removed RLM-internal branding from model-visible prompts, richer metrics, hardened root-tool transport (no unsafe pickle), and a reworked harness install flow that runs from a uv workspace checkout.
- Runtime performance and reliability improvements including executor autoscaling, incremental metrics, threaded file I/O, event loop lag monitoring, multi-worker env server support, GC tuning before accepting requests, `setproctitle` labels, dead-tunnel auto-recovery in `CliAgentEnv`, and safer task cancellation paths.
- Richer `vf-tui` with a log viewer, run comparison mode, toggleable markdown/reasoning rendering, rollout and unique-prompt counts with responsive layout, and saved-state columns in the info view.
- Expanded evaluation ergonomics with configurable `output_dir`, `[[ablation]]` model/endpoint overrides, `max_total_tokens` for `MultiTurnEnv`, `extra_headers_from_state` and `headers`/`extra_headers` support in `endpoints.toml`, `X-Session-ID` for DP-aware routing, preserved multimodal media in saved results, and exported eval parser/normalization helpers for Prime CLI reuse.
- New Hosted Evaluations docs plus an environment performance guide, refreshed BrowserEnv README, and updated Secrets/Hub guidance across docs and agent skills.

## Changes included in v0.1.12 (since v0.1.11)

### Clients, RLM, and rollout execution

- feat: composable Task/Agent/Environment architecture (#1067)
- upstream `RlmComposableEnv` into `ComposableEnv`/`TaskSet`/`Harness` (#1158)
- move harnesses (rlm, opencode) and tasksets (swe, lean, math, cp, harbor) into `verifiers.envs.experimental.composable` (#1131)
- RLM: `RLMPromptBuilder` (#1070)
- RLM: context dropping & summarization (#1072)
- replace `remove_conversation_turns` with `summarize_turns` standard tool (#1095)
- add `max_turns_in_context`, fix answer extraction, document metrics (#1099)
- RLM: inform model about `max_turns_in_context` limit in scaffolding (#1111)
- remove RLM branding from model-visible prompts and messages (#1089)
- change `tools` arg to pass standard tools to root LLM (#1087)
- add `enable_sub_llms` toggle to `RLMEnv` (#1085)
- simplify RLM message transcript handling (#1116)
- RLM: improve prompts and metrics (#1102)
- refactor: rename RLM metrics for consistency (#1086)
- remove token/timing info from `llm_batch` output and add `max_turns` metric (#1098)
- replace timing info in RLM REPL output with root tool time metrics (#1097)
- harden RLM root-tool transport to remove unsafe pickle deserialization (#1104)
- RLM: remove dead code, harden tunnels (#1107)
- run RLM harness from a uv workspace checkout (#1139)
- revert inline install, use rlm's `install.sh` (#1144)
- clone via git protocol instead of fetching `install.sh` (#1159)
- update RLM harness test to match git-clone install script (#1160)
- RLM harness: install from arbitrary branch (#1153)
- fix RLM harness to use per-example `AGENT_WORKDIR` (#1143)
- port rlm harness dedup install script fix (#1133)
- set `RLM_KERNEL_PYTHON` to sandbox `.venv` for inline imports (#1145)
- guard `RLM_KERNEL_PYTHON` on successful ipykernel install (#1150)
- pin `ipykernel<7` for older sandbox Pythons (#1151)
- fix RLM bash timeout (#1079)
- fix: handle `CommandTimeoutError` in `RLMEnv` (#1069)
- deprecate `RolloutGatewayMixin` (#1017)
- add `NeMoRLChatCompletionsClient` for NeMo Gym model servers (#1141)
- feat: send `X-Session-ID` header during eval for DP-aware routing (#1137)
- feat: add `extra_headers_from_state` to `ClientConfig` (#1048)
- fix: handle `None` prompt/completion token ids in `parse_tokens` (#1066)
- fix tool args passing (#1106)
- fix TITO in opencode envs: bridge extraction, truncation gate, tool-call handling (#1005)
- fix: remove content `rstrip` in `normalize_response` to preserve TITO prefix match (#1081)

### Env server, sandbox, and runtime reliability

- feat: multi env worker (#1055)
- fix: propagate `json_logging` to env workers (#1138)
- tune GC on env server before accepting requests (#1022)
- feat: set process titles on env server and workers (#1082)
- perf: executor autoscaling (#1039)
- perf: incremental metrics (#1036)
- perf: offload file I/O to thread pool (#1037)
- feat: improve event loop lag monitor (#1038)
- fix `get_free_port_pair()` TOCTOU race condition (#1013)
- fix: task cancellation race + RLM sandbox workers (#1035)
- fix: call `uncancel()` after catching `CancelledError` in `process_request` (#1047)
- fix cancelled + serialize error (#1044)
- detect server-side tunnel death and auto-recreate in `CliAgentEnv` (#1127)
- fix `AgentError` double-wrapping in `poll_job_completion` (#1130)
- fix: clear root logger handlers hijacked by swebench import (#1163)
- use SDK read-file endpoint and bg job handling in `SandboxMixin` (#1084)

### Evaluation UX, metrics, and configuration

- `vf-tui`: log viewer (#1075)
- `vf-tui`: fixes & features (including comparison mode and markdown/reasoning toggles) (#1007)
- `vf-tui`: show rollouts and unique prompts, better dynamic width (#1060)
- show saved state columns in TUI info view (#1091)
- make `output_dir` configurable in evals (#1029)
- handle ablation `model` and `endpoint_id` overrides (#1135)
- export eval parser and normalization helpers for Prime CLI reuse (#1135)
- feat: add `max_total_tokens` parameter to `MultiTurnEnv` (#1101)
- support `headers`/`extra_headers` in `endpoints.toml` (#1051)
- preserve multimodal media in saved eval results (#1015)
- fix display of custom sampling args (#1025)
- fix output dir logging (#1041)
- fix host-side eval in composable CP wrapper parsing (#1165)
- fix composable `mkdir` path quoting (#1110)

### Environments, multimodality, and integrations

- add `BrowserEnv` integration README (#1020)
- `opencode_rlm_env` (#1023)
- misc improvements to opencode envs (#999)
- perf improvs for opencode envs + math rubric (#1034)
- opencode envs (including `CliAgentEnv` hardening, hybrid math rubric overhaul, and log capture) (#1005)
- fix: revert `opencode_env` config regression and move RLM logic out of `cli_agent_env` (#1042)
- fix opencode config for model names without slash (#1114)
- feat: dataset builder pattern for lazy loading in all environments (#1064)
- add cleanup and teardown lifecycle hooks to `Rubric` (#1026)
- remove redundant msg normalization + align `env_response` API (#1027)
- chore: reuse math rubric in hybrid math rubric (#1043)
- perf: math rubric skip overlong answers (#1046)
- fix math rubric timeout (#1096)
- lazily import packages (#1019)
- fix: env tests (#1061)

### Docs, CLI, and tooling

- docs: add performance guide for environments (#1045)
- add hosted evaluations section to eval docs (#1040)
- update Secrets guidance (BrowserBase README) (#1056)
- docs: prefix `prime eval` models (#1125)
- `tomllib`/`tomli` guard for Python 3.10 (#1136)
- pin `regex<2026.4.4` (missing cp312/cp313 wheels) (#1109)
- pin uv `<0.11.0` to fix flash-attn resolution (#1057)
- bump uv requirement to `>=0.11.1` (#1112)
