# Verifiers v0.1.15.dev8 Release Notes

*Date:* 05/22/2026

## Highlights since v0.1.15.dev7

- **v1 Taskset/Harness loading is stricter and more package-native.** v1 configs now keep taskset and harness child config objects, package component loading is explicit, and the BYO Harness docs, generated agent guidance, and create-environments skill describe reusable tasksets and harnesses as the golden path.
- **Eval configuration defaults are cleaner.** TOML-backed eval runs save results by default, sampling sections forward reasoning kwargs to clients, and stale pre-eval install docs were removed.
- **Rollout and client behavior is more robust.** SWE rollouts with sandbox errors skip scoring, save deltas reset cleanly on non-monotonic trajectories, renderer overlong-prompt failures normalize to `vf.OverlongPromptError`, and routed expert responses can carry sidecar data.
- **Developer tooling is tighter.** The eval viewer TUI was modernized, the legacy `vf-tui` entrypoint was retired, renderer client error handling was tightened, and the router replay performance experiment was reverted.

## Changes included in v0.1.15.dev8 (since v0.1.15.dev7)

### v1 Taskset/Harness

- Rework v1 harness and taskset config classes (#1392)
- Add strict package loaders for v1 tasksets and harnesses (#1429)

### Evals and rollouts

- Modernize eval viewer TUI (#1393)
- Forward reasoning kwargs from eval TOML sampling sections (#1404)
- Skip SWE sandbox scoring for errored rollouts (#1412)
- Support routed experts response sidecar (#1423)
- Remove stale pre-eval `prime env install` docs (#1434)
- Default TOML eval runs to save results (#1435)

### Runtime and client fixes

- chore: bump renderers to 0.1.8.dev2 (supersedes #1366) (#1395)
- fix(save_utils): reset delta baseline on non-monotonic trajectories (#1400)
- fix(renderer-client): translate renderers.OverlongPromptError into vf.OverlongPromptError and require renderers>=0.1.8.dev4 (#1408)
- [Router Replay]: Improve performance by removing Pydantic validation (#1394)
- Revert "[Router Replay]: Improve performance by removing Pydantic validation" (#1422)

### Tooling and docs

- Retire vf-tui entrypoint (#1398)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev7...v0.1.15.dev8
