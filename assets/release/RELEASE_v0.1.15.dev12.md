# Verifiers v0.1.15.dev12 Release Notes

*Date:* 05/29/2026

## Highlights since v0.1.15.dev11

- **Renderer trajectories carry richer token provenance.** Renderer-backed clients now preserve per-token prompt attribution, serialize rendered-token payloads cleanly across the v1 boundary, route `chat_template_kwargs` through `RendererConfig`, and surface every parsed tool call from renderer responses.
- **v1 sandbox execution is more production-ready.** Sandbox-backed programs use the managed Python runtime, attach v1 sandbox labels, reuse sandbox clients correctly, report upload errors clearly, and handle routed-expert replay starts for bridged prompts.
- **Reusable v1 packages and tasksets are staged for broader use.** Standalone `tasksets` and `harnesses` packages are in place with hardened program config handling, and the Lean taskset now matches the final `LAST EXIT_CODE` marker to close a reward bypass.

## Changes included in v0.1.15.dev12 (since v0.1.15.dev11)

### Renderer and trajectory plumbing

- feat(renderer-client): give per-token prompt attribution to TrajectoryStep (#1414)
- fix(renderer-client): surface every parsed tool_call, drop status filter (#1463)
- Pipe chat_template_kwargs into typed RendererConfig (#1468)
- renderers <-> v1: serializable RenderedTokens (#1471)

### v1 sandbox runtime

- Use managed sandbox Python runtime (#1450)
- Set routed experts replay start for bridged prompts (#1466)
- Add v1 sandbox labels (#1476)
- Fix v1 sandbox client reuse and upload errors (#1483)

### Reusable v1 packages and tasksets

- fix(lean-taskset): match LAST EXIT_CODE marker to close reward bypass (#1480)
- Stage standalone tasksets and harnesses packages with hardened v1 program configs (#1475)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev11...v0.1.15.dev12
