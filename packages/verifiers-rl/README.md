# verifiers-rl

Optional RL trainer package for `verifiers`.

Install:

```bash
uv add verifiers-rl
```

This package provides:

- `vf-rl`
- `vf-train`
- `vf-vllm`
- `verifiers_rl.rl` (RLTrainer implementation)

`verifiers` core remains usable without this package.

## First-time PyPI registration (same style as `verifiers`)

The easiest path is to use the same token-based `uv publish` flow used by core `verifiers`:

1. Create a PyPI API token with permission to publish `verifiers-rl`.
2. Add it as a GitHub secret (you can reuse the existing `PYPI_TOKEN` used for core releases).
3. Build + publish this package from the subdirectory:

```bash
uv build packages/verifiers-rl
uv publish --token "$PYPI_TOKEN" packages/verifiers-rl/dist/*
```

The first successful upload registers the `verifiers-rl` project name on PyPI (if available).
