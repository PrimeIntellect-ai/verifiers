# Verifiers v0.1.12.dev1 Release Notes

*Date:* 03/25/2026

## Highlights since v0.1.12.dev0

- Multi-environment worker support for parallel environment execution.
- Dataset builder pattern enabling lazy loading across all environments.
- Pinned uv <0.11.0 to fix flash-attn dependency resolution.
- Environment test fixes.

## Changes included in v0.1.12.dev1 (since v0.1.12.dev0)

### Features

- feat: multi env worker (#1055)
- feat: dataset builder pattern for lazy loading in all environments (#1064)

### Fixes and maintenance

- fix: env tests (#1061)
- Pin uv <0.11.0 to fix flash-attn resolution (#1057)
- Update BrowserBase README (#1056)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.12.dev0...v0.1.12.dev1
