# Verifiers v0.1.15.dev6 Release Notes

*Date:* 05/14/2026

## Highlights since v0.1.15.dev5

- **Routed-experts token metadata is handled as an opaque payload.** OpenAI-compatible chat and text-completions clients now preserve routed-experts data without expanding it through user-facing message types, and token parsing truncates that sidecar alongside prompt/completion tokens when a rollout is clipped to `max_seq_len`.
- **v1 `EnvConfig` typing is stricter.** The v1 loader envelope, config docs, generated environment templates, examples, and tests now keep environment-level configuration narrow while routing taskset- and harness-owned fields through their concrete config classes.

## Changes included in v0.1.15.dev6 (since v0.1.15.dev5)

### Tokens and clients

- Consume split routed-experts payloads in OpenAI-compatible clients (#1363)
- Move routed-experts helpers into response utilities (#1373)

### v1 config

- Tighten v1 `EnvConfig` typing (#1371)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev5...v0.1.15.dev6
