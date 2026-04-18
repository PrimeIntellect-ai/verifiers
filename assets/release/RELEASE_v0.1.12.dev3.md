# Verifiers v0.1.12.dev3 Release Notes

*Date:* 04/06/2026

## Highlights since v0.1.12.dev2

- Added `max_total_tokens` parameter to MultiTurnEnv for controlling total token budget.
- Introduced `max_turns_in_context` and `summarize_turns` standard tool for smarter context management in RLM.
- Hardened RLM root-tool transport by removing unsafe pickle deserialization.
- Improved RLM prompts, metrics, and scaffolding with better turn-limit awareness.
- Added support for `headers`/`extra_headers` in `endpoints.toml`.

## Changes included in v0.1.12.dev3 (since v0.1.12.dev2)

### Features and enhancements

- feat: add `max_total_tokens` parameter to MultiTurnEnv (#1101)
- feat: add `max_turns_in_context` with answer extraction and metric docs (#1099)
- feat: replace `remove_conversation_turns` with `summarize_turns` standard tool (#1095)
- feat: support `headers`/`extra_headers` in `endpoints.toml` (#1051)
- feat: RLM inform model about `max_turns_in_context` limit in scaffolding (#1111)
- feat: replace timing info in RLM REPL output with root tool time metrics (#1097)
- feat: remove token/timing info from `llm_batch` output and add `max_turns` metric (#1098)

### Fixes and maintenance

- fix: opencode config for model names without slash (#1114)
- fix: composable `mkdir` path quoting (#1110)
- fix: tool args passing (#1106)
- fix: math rubric timeout (#1096)
- fix: pin `regex<2026.4.4` for missing cp312/cp313 wheels (#1109)
- fix: bump uv requirement to `>=0.11.1` (#1112)

### Security and hardening

- security: harden RLM root-tool transport to remove unsafe pickle deserialization (#1104)
- refactor: RLM remove dead code, harden tunnels (#1107)
- refactor: RLM improve prompts and metrics (#1102)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.12.dev2...v0.1.12.dev3
