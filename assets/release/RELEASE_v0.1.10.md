# Verifiers v0.1.10 Release Notes

*Date:* 02/10/2026

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.9...v0.1.10

## Highlights since v0.1.9

- Expanded environment support with OpenEnv and BrowserEnv integrations, env worker plumbing, and continued improvements to `CliAgentEnv`/`RLMEnv` sandbox reliability and customization hooks.
- Upgraded evaluation ergonomics with resumed evals, improved TUI info/log presentation, better rollout/token tracking, and non-TUI overflow rendering fixes.
- Improved reliability across model/runtime boundaries with timeout hardening, safer sandbox lifecycle behavior, and richer error/metadata handling.
- Modernized workspace setup and contributor workflows (`vf-setup` endpoint/config updates, GEPA config support, Prime CLI refactor, skills scaffolding, and AGENTS guidance updates).
- Added opencode harbor enhancements, including TITO support, tunnel sync stop behavior, and terminal-bench task coverage.

## Changes included in v0.1.10 (since v0.1.9)

### Environment, rollout, and runtime improvements

- openenv integration (#829)
- Add Browser Env Integration (#732)
- resume evals (#803)
- add Client Pool (#815)
- RLM: Eager sandbox creation, conditional pip install (#834)
- RLM: Add RLMEnv sandbox hooks for safer customization (#849)
- RLM: Make FIFO IO non-blocking (#850)
- `CliAgentEnv`: add `SandboxMixin`, refactor `InterceptionServer` (#847)
- rlm: migrate sandbox executor to SandboxMixin (#875)
- env worker integration (#832)
- track vf + env version in metadata (#881)
- handle empty metrics (#855)
- move sanitize_metadata out of save_metadata (#852)
- improve env client timeouts (#872)
- Fix ty logger protocol typing in sandbox retry setup (#835)
- Fix vf-eval concurrent rollout label to use effective cap (#836)

### Evaluation UX, logging, and metrics

- Add robust token usage tracking (#858)
- Tighten vf-tui info preview formatting and typing checks (#830)
- Add subtle `--debug` hint beneath Logs panel (#824)
- Fix vf-eval non-TUI live overflow rendering (#883)
- misc logging improvs (#882)

### Setup, CLI, and configuration

- vf-setup: prefer endpoints.toml, rename configs/lab->configs/rl, add GEPA configs, deprecate --vf-rl (#859)
- Support long endpoint field names in TOML registries (#861)
- prime CLI refactor (vf) (#870)
- refactor: split RL trainer into optional in-repo verifiers-rl package (#843)
- move rlm secrets out of vf and into research-environments (#856)

### Documentation, workflows, and skills

- Compile AGENTS docs from modular assets and make guidance concrete (#857)
- skills setup (#873)
- Strengthen lab AGENTS env-development guardrails (#876)
- Clarify MCPEnv is for global read-only MCP servers (#838)
- docs: remove parser-centric guidance from environment READMEs (#839)
- docs: remove parser field from env init README template (#840)
- chore: enforce ruff formatting and improve dev tooling docs (#845)

### Integrations and environment packages

- openenv: default template proj/ path and simplify prompt renderer signatures (#853)
- remove vf pin in `opencode_harbor` (#844)
- opencode env: TITO support, tunnel sync stop, add terminal-bench (#874)
- ci: skip terminus_harbor in test-envs (#846)
- fix math rubric timeouts (#831)
- Fix for dir resolutions (#879)
- Update browse-environments freshness and quality priorities (#884)
- Clarify agent skill handling (#886)
