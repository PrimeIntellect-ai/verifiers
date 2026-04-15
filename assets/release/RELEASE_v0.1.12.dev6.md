# Verifiers v0.1.12.dev6 Release Notes

*Date:* 04/15/2026

## Highlights since v0.1.12.dev5

- Exported the eval parser and normalization helpers so Prime CLI can reuse Verifiers eval parsing and config loading.
- Fixed eval ablation config overrides so explicit `model` and `endpoint_id` settings behave correctly.
- Propagated `json_logging` from EnvServer to env workers so worker logs stay structured when JSON logging is enabled.
- Fixed the RLM harness install flow to avoid duplicate installs and restored Harbor TOML loading on Python 3.10.

## Changes included in v0.1.12.dev6 (since v0.1.12.dev5)

### Features and enhancements

- feat: export eval parser and normalization helpers for Prime CLI reuse (#1135)
- fix: handle ablation `model` and `endpoint_id` overrides in eval configs (#1135)

### Fixes and maintenance

- fix: propagate `json_logging` to env workers (#1138)
- fix: port RLM harness dedup install script update (#1133)
- fix: add `tomli` fallback for Harbor on Python 3.10 (#1136)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.12.dev5...v0.1.12.dev6
