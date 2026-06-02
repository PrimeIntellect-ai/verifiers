# Verifiers v0.1.15.dev18 Release Notes

*Date:* 06/01/2026

## Highlights since v0.1.15.dev17

- **v1 interception and sandbox config.** The interception server now accepts request bodies up to 24 MB, and the v1 `SandboxConfig` regains its `labels` field.
- **Standalone packages and agents.** The `tasksets` and `harnesses` packages move to OIDC trusted publishing, and the mini-swe-agent harness issues real parallel tool calls.

## Changes included in v0.1.15.dev18 (since v0.1.15.dev17)

### v1 runtime and interception

- Re-add labels field to v1 SandboxConfig (#1512)
- Raise interception server request body limit to 24 MB (#1504)

### Harnesses and packaging

- Use real parallel tool calls for mini-swe-agent (#1505)
- Set trusted publishing for tasksets and harnesses (#1508)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev17...v0.1.15.dev18
