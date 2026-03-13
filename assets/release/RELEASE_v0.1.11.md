# Verifiers v0.1.11 Release Notes

*Date:* 03/12/2026

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.10...v0.1.11

## Highlights since v0.1.10

- Unified the client stack around shared client abstractions, with better multimodal handling, provider defaults, token accounting, and more consistent response/error behavior.
- Reworked `RLMEnv` and rollout execution with sandbox-only execution, shared interception pools, `RolloutGatewayMixin`, tunnel reuse and recovery, and clearer model message history and debug logs.
- Improved evaluation ergonomics with heartbeat monitoring, a refined multi-environment TUI, overview and settings panels, abbreviated summaries, richer detail headers, and cleaner Rich log rendering.
- Hardened env server and sandbox reliability with spawn-based sidecars, deadlock and crash recovery, client cancellation propagation, safer teardown and cleanup paths, optional env server opt-out, and configurable HTTP connect timeouts.
- Added new evaluation and environment capabilities including pass@k and pass^k metrics, router replay, `[[ablation]]` eval sweep syntax, single-file environment imports, and bundled opencode environments and utilities.

## Changes included in v0.1.11 (since v0.1.10)

### Clients, RLM, and rollout execution

- unified client interface (#897)
- Remove local execution backend from `RLMEnv`, always use sandbox (#905)
- migrate `RLMEnv` to unified client types (#914)
- move token utils into the OpenAI token client (#913)
- provider defaults (#931)
- `RLMEnv`: add shared interception pool for tunnel reuse (#939)
- `RLMEnv`: `.messages` contains the model's message history (#946)
- add `RolloutGatewayMixin` for server-side rollout execution (#954)
- `RLMEnv`: simplify constructor and internals (#966)
- `RLMEnv`: `root_max_completion_tokens` support (#973)
- remove `get_model_response` override from `RLMEnv` (#977)
- `RLMEnv`: better debug logs (#1009)
- Support interleaved rollouts with `include_sub_llm_in_trajectory=True` (#900)
- deprecate `interleaved_rollouts` (#912)
- do not set strict mode if not set (#915)
- do not set strict token-completions mode by default (#911)
- Add `SubLLMEmptyModelResponseError` and exception hierarchy tests (#894)

### Env server, sandbox, and runtime reliability

- Default `interception_port` to `0` to avoid port collisions (#906)
- use spawn multiprocessing method for sidecar env server (#917)
- fix env server deadlock (#921)
- env server recovery (#927)
- add `json_logging` support to env server (#930)
- fix: tokenize `env_response` with full conversation context (#938)
- optional opt-out of env server in evals (#945)
- fix: prevent silent message loss on the ZMQ socket (#951)
- fix browser sandbox leak by wrapping errors in a `vf.Error` subclass (#958)
- restart dead tunnels (#964)
- fix: `SandboxMixin` logger not inheriting verifiers log config (#970)
- fix empty `SandboxError` message in the RLM executor (#971)
- propagate client-side cancellation to the env server (#974)
- fix: prevent CUA sandbox leaks during setup (#983)
- make `httpx` connect timeout configurable (#1006)

### Evaluation UX, metrics, and configuration

- add heartbeat monitoring to `vf-eval` (#904)
- add pass@k and pass^k metrics with configurable threshold (#944)
- add router replay (#923)
- Add `[[ablation]]` sweep syntax for eval TOML configs (#993)
- Add overview panel and collapse non-running envs in the `vf-eval` display (#978)
- multi-env `vf-eval`: switch between progress bars with arrow keys (#979)
- Remove manual log wrapping and let Rich handle it (#985)
- sanitize illegible binary characters in Rich display (#986)
- `vf-eval`: add settings panel in summary and `--abbreviated-summary` flag (#987)
- Show `env_args` in the detail panel header (#1001)
- Refine eval TUI (#1000)

### Environments, multimodality, and integrations

- use MITO for multimodal (#956)
- make multimodal detection more robust (#972)
- also check for reasoning before raising error (#918)
- Handle single-file env import (#959)
- add opencode utils (#969)
- opencode envs (#984)
- fix TITO in `CliAgentEnv` (#991)
- perf improvements to `TextArenaEnv` and Wordle (#943)

### Docs, CLI, and tooling

- Update `README.md` paths and environment documentation links (#898)
- replace outdated `vf-*` CLI references with `prime` CLI commands (#926)
- Sync `endpoints.toml` API type fields with `endpoints.py` (#925)
- Change `publish-envs` workflow to manual and release triggers (#922)
- Fix all ty type check errors in `verifiers/` (#998)
- pin ty version because `0.0.22` is buggy (#1014)
- Stabilize dataset map audio test (#1003)
