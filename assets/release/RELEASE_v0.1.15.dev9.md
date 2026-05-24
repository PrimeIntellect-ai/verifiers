# Verifiers v0.1.15.dev9 Release Notes

*Date:* 05/22/2026

## Highlights since v0.1.15.dev8

- **v1 TextArena tasksets are now package-native.** `TextArenaTaskset` and `TextArenaTasksetConfig` are exported from the public v1 surfaces, and `wordle-v1` is bundled as a concrete TextArena-backed example with task-owned game mechanics, rewards, and eval defaults.
- **v1 environment loading is simpler and stricter.** `load_environment` now derives taskset and harness config types from module loader annotations, validates child config objects at the boundary, and keeps package loading out of `EnvConfig` and `vf.Env` construction.
- **Tool, harness, and API cleanup tightened the surface.** Callable tool schemas now propagate strict schema metadata, the Pi harness uses the Earendil package, and unused verifier helper APIs were removed.

## Changes included in v0.1.15.dev9 (since v0.1.15.dev8)

### v1 Tasksets and environment loading

- Migrate textarena_taskset into the v1 tasksets package (#1446)

### Tools and harnesses

- Propagate strict callable tool schemas (#1445)
- Use Earendil Pi package (#1442)

### API cleanup

- Remove unused verifier helpers (#1437)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev8...v0.1.15.dev9
