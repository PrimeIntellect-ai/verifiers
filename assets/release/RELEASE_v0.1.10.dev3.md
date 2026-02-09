# Verifiers v0.1.10.dev3 Release Notes

*Date:* 02/08/2026

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.9...v0.1.10.dev3

## Highlights since v0.1.9

- Added new environment capabilities, including **OpenEnv integration**, **BrowserEnv integration**, and **env server** support for more flexible tool and environment workflows.
- Expanded evaluation UX with **eval TUI**, copy mode, improved logs/debug display, rollout token usage tracking, and richer saved-output rendering for tool calls.
- Introduced and iterated on **RLMEnv** improvements: tool partitioning (`tools`, `root_tools`, `sub_tools`), better stop/error propagation, prompt/verbosity controls, safer sandbox lifecycle handling, and new sandbox hooks for customization.
- Improved reliability across execution and infrastructure paths via retries for infrastructure and model-response errors, better auth/overlong prompt handling for OpenRouter, and cleanup fixes to avoid task/sandbox leakage.
- Modernized setup and training ergonomics with `vf-setup` config changes (`endpoints.toml`, `configs/rl`, GEPA configs), support for long TOML endpoint field names, and an optional in-repo `verifiers-rl` package split.
- Hardened runtime internals with `CliAgentEnv` sandbox/interception refactors, client pooling, non-blocking FIFO IO for RLM, and metadata/metrics handling fixes.
- Added broader OpenEnv ecosystem support and examples (e.g., `openenv_echo`, `openenv_textarena`, `opencode_harbor`) with updated version requirements.

## Incremental changes since v0.1.10.dev2

- Compile AGENTS docs from modular assets and make guidance concrete (#857)
- vf-setup: prefer endpoints.toml, rename configs/lab->configs/rl, add GEPA configs, deprecate --vf-rl (#859)
- Support long endpoint field names in TOML registries (#861)
- Add robust token usage tracking (#858)
- move rlm secrets out of vf and into research-environments (#856)
- `CliAgentEnv`: add `SandboxMixin`, refactor `InterceptionServer` (#847)
- handle empty metrics (#855)
- move sanitize_metadata out of save_metadata (#852)
- openenv: default template proj/ path and simplify prompt renderer signatures (#853)
- refactor: split RL trainer into optional in-repo verifiers-rl package (#843)
- add Client Pool (#815)
- chore: enforce ruff formatting and improve dev tooling docs (#845)
- RLM: Make FIFO IO non-blocking (#850)
- RLM: Add RLMEnv sandbox hooks for safer customization (#849)
- RLM: Eager sandbox creation, conditional pip install (#834)
- ci: skip terminus_harbor in test-envs (#846)
- resume evals (#803)
- remove vf pin in `opencode_harbor` (#844)
- fix math rubric timeouts (#831)
- docs: remove parser-centric guidance from environment READMEs (#839)
- openenv integration (#829)
- Fix ty logger protocol typing in sandbox retry setup (#835)
- docs: remove parser field from env init README template (#840)
- Clarify MCPEnv is for global read-only MCP servers (#838)
- Fix vf-eval concurrent rollout label to use effective cap (#836)
- Tighten vf-tui info preview formatting and typing checks (#830)
- Add subtle `--debug` hint beneath Logs panel (#824)
