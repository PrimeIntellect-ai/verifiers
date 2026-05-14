# Verifiers v0.1.15.dev1 Release Notes

*Date:* 05/12/2026

## Highlights since v0.1.15.dev0

- **Renderer-native multimodal transport.** Threads renderer-emitted `multi_modal_data` through rollout generation, trajectory tokens, msgpack transport, and save output, with per-step delta encoding to avoid duplicating image payloads across rollout windows.
- **Provider-specific reasoning controls.** Maps `reasoning_effort` correctly for Anthropic models across native Anthropic calls and OpenAI-compatible Anthropic routes, including adaptive thinking defaults for newer Claude models.
- **Runtime and release hardening.** Removes the fixed ZMQ env request timeout, tightens v1 lifecycle handler discovery, and adds BugBot rules that keep release metadata and dependency sources publishable.

## Changes included in v0.1.15.dev1 (since v0.1.15.dev0)

### Features and enhancements

- feat(renderer-client): thread multimodal sidecar through rollout + transport (#1346)
- Map Anthropic reasoning effort by provider (#1338)

### Fixes and maintenance

- Fix ZMQ env timeouts (#1355)
- Fix v1 lifecycle handler discovery (#1347)
- Strengthen BugBot release and dependency rules (#1354)
- Tighten BugBot releasability and dependency rules (#1353)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev0...v0.1.15.dev1
