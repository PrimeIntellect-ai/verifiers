# Verifiers v0.1.15.dev10 Release Notes

*Date:* 05/25/2026

## Highlights since v0.1.15.dev9

- **v1 environment authoring has clearer component loading.** `prime env init` is back to a v0 stub by default, `--v1` emits the thin Taskset/Harness loader templates, and public `vf.load_taskset` / `vf.load_harness` helpers now handle annotation-driven config coercion.
- **Serve and eval routing paths are smoother under load.** Env worker/server CPU-heavy work moves off the event loop with scaled executors and uvloop support, `EnvServer` now installs the scaled executor correctly, renderer requests preserve state-derived headers, and eval sticky routing defaults to per-trajectory sessions.
- **Harness and client behavior got targeted fixes.** RLM skills upload to the path the harness reads, generated MCP setup scripts are POSIX `sh` compatible, Anthropic mixed text/image conversion preserves unsupported image markers, and duplicate/single-use helper code was simplified.

## Changes included in v0.1.15.dev10 (since v0.1.15.dev9)

### v1 loaders and harnesses

- [codex] Revise v1 init taskset loaders (#1449)
- Fix RLM skills upload path (#1451)
- Make harness setup scripts POSIX sh compatible (#1457)

### Serve, eval, and renderer runtime

- perf(serve): reduce worker/router event-loop lag (#1453)
- Fix EnvServer executor installation (#1455)
- Use renderers offload helper in renderer client (#1452)
- [codex] Forward renderer request headers (#1459)
- [codex] Default eval session header to trajectory_id (#1460)

### Client behavior and cleanup

- Inline single-use verifier helpers (#1443)
- Fix Anthropic unsupported image markers (#1461)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev9...v0.1.15.dev10
