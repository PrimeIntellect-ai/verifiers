# Verifiers v0.1.15.dev11 Release Notes

*Date:* 05/25/2026

## Highlights since v0.1.15.dev10

- **v1 taskset authoring has a tighter loader contract.** Taskset configs now use clearer task source fields, task inputs normalize at the boundary, taskset loaders are registry-backed, and environment loaders route through taskset hooks instead of class defaults.
- **v1 examples and docs match the new taskset shape.** Bundled environments, README examples, BYO Harness docs, and generated environment guidance now use the `load_taskset(config)` / `load_environment(config)` pattern consistently.
- **Renderer configuration now uses the typed config surface.** Renderer clients consume `RendererConfig` directly, keeping renderer request configuration aligned with the shared typed contract.

## Changes included in v0.1.15.dev11 (since v0.1.15.dev10)

### v1 taskset authoring

- Rework TasksetConfig fields + loader patterns (#1462)

### Renderer configuration

- Consume the typed RendererConfig surface (#1467)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev10...v0.1.15.dev11
