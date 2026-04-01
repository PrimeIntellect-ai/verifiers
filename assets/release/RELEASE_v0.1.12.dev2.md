# Verifiers v0.1.12.dev2 Release Notes

*Date:* 04/01/2026

## Highlights since v0.1.12.dev1

- Added major composability improvements for task/agent/environment architecture.
- Improved execution ergonomics by routing standard tools through the root LLM via the `tools` argument.
- Strengthened TUI observability with saved-state column visibility updates.
- Removed RLM branding from model-visible prompts/messages and aligned RLM metrics naming for consistency.
- Hardened token parsing behavior for `None` prompt/completion token IDs.

## Changes included in v0.1.12.dev2 (since v0.1.12.dev1)

### Features and architecture

- feat: composable Task/Agent/Environment architecture (#1067)
- feat: change `tools` arg to pass standard tools to root LLM (#1087)
- Show saved state columns in TUI info view (#1091)

### Fixes and maintenance

- fix: handle None prompt/completion token ids in parse_tokens (#1066)
- refactor: rename RLM metrics for consistency (#1086)
- remove RLM branding from model-visible prompts and messages (#1089)
- feat: add enable_sub_llms toggle to RLMEnv (#1085)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.12.dev1...v0.1.12.dev2
