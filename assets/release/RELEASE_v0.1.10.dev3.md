# Verifiers v0.1.10.dev3 Release Notes

*Date:* 02/07/2026

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.9...v0.1.10.dev3

## Highlights since v0.1.9

- Added new environment capabilities, including **OpenEnv integration**, **BrowserEnv integration**, and **env server** support for more flexible tool and environment workflows.
- Expanded evaluation UX with **eval TUI**, copy mode, improved logs/debug display, rollout token usage tracking, and richer saved-output rendering for tool calls.
- Introduced and iterated on **RLMEnv** improvements: tool partitioning (`tools`, `root_tools`, `sub_tools`), better stop/error propagation, prompt/verbosity controls, safer sandbox lifecycle handling, and new sandbox hooks for customization.
- Improved reliability across execution and infrastructure paths via retries for infrastructure and model-response errors, better auth/overlong prompt handling for OpenRouter, and cleanup fixes to avoid task/sandbox leakage.
- Added broader OpenEnv ecosystem support and examples (e.g., `openenv_echo`, `openenv_textarena`, `opencode_harbor`) with updated version requirements.

## Incremental changes since v0.1.10.dev2

- RLM: Add RLMEnv sandbox hooks for safer customization (#849)
- RLM: Eager sandbox creation, conditional pip install (#834)
- ci: skip terminus_harbor in test-envs (#846)
- remove vf pin in `opencode_harbor` (#844)
- docs: remove parser-centric guidance from environment READMEs (#839)
- openenv integration (#829)
- Fix ty logger protocol typing in sandbox retry setup (#835)
- docs: remove parser field from env init README template (#840)
- Clarify MCPEnv is for global read-only MCP servers (#838)
- Fix vf-eval concurrent rollout label to use effective cap (#836)
- Tighten vf-tui info preview formatting and typing checks (#830)
- Add subtle `--debug` hint beneath Logs panel (#824)
