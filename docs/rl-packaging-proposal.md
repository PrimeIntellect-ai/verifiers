# RL packaging proposal: move `RLTrainer` out of core library surface

## Problem statement

Today `RLTrainer` and its implementation live under `verifiers/rl/`, while the core package (`verifiers`) still exports RL symbols and scripts (`vf-rl`, `vf-train`, `vf-vllm`). This causes a few practical issues:

- Core development workflows and type checking regularly traverse RL modules with heavyweight GPU dependencies.
- The RL codepath is demo/educational and not actively maintained, but appears as a first-class part of core.
- The current structure encourages accidental imports of RL modules in CPU-only environments.

## Goals

- Keep RL code in this repo.
- Preserve current user workflow (`vf-rl` entrypoint and existing config style).
- Make RL installable on demand, with minimal impact on core users.
- Localize imports so non-RL installs do not import RL modules.

## Recommended approach (phased)

### Phase 1: Extract RL into a workspace subpackage

Create a dedicated package directory in-repo:

- `packages/verifiers-rl/`
  - `pyproject.toml` (`name = "verifiers-rl"`)
  - `verifiers_rl/` (code moved from `verifiers/rl/` and RL scripts)

Key points:

- Keep `verifiers` as a dependency of `verifiers-rl`.
- Move RL-only CLIs into `verifiers-rl`:
  - `vf-rl`
  - `vf-train`
  - `vf-vllm`
- Keep thin compatibility wrappers in `verifiers` for one deprecation cycle.

### Phase 2: Add optional compatibility shims in core

For one deprecation cycle, keep tiny compatibility modules in core:

- `verifiers/rl/*` modules proxying to `verifiers_rl` with clear install hints.
- `verifiers.__getattr__` entries for `RLTrainer` etc. with install guidance:
  - "Install `verifiers-rl` and import from `verifiers_rl` (or use `vf-rl`)."

### Phase 3: Fully decouple

After deprecation window:

- Remove RL compatibility exports/modules from core.
- Keep docs pointing to `uv add verifiers-rl`.

## Packaging and installation UX

Target UX:

```bash
uv add verifiers
# no RL deps pulled

uv add verifiers-rl
# installs RL trainer + heavy deps + vf-rl scripts
```

For editable repo development:

```bash
uv sync
uv sync --package verifiers-rl
```

(Exact command shape depends on final workspace tool choice.)

## Import-localization guidelines

Inside `verifiers-rl`:

- Keep heavy imports (`torch`, `vllm`, `deepspeed`, `flash_attn`) function-local where practical.
- Avoid top-level imports of training backends in modules used only for CLI argument parsing or config loading.
- Keep type-check-only imports behind `if TYPE_CHECKING:` where possible.

Inside core `verifiers`:

- No imports from `verifiers_rl` at module import time.
- Compatibility paths should fail loudly with an install instruction if plugin is absent.

## PyPI registration and release automation

### Do we need separate PyPI registration?

Yes. `verifiers-rl` is a distinct distribution name from `verifiers`, so it needs its own project on PyPI (first upload creates it if the name is available and the publisher is authorized).

### Can this be done programmatically in first release?

Yes, mostly:

1. Configure Trusted Publishing for this GitHub repo in PyPI for project `verifiers-rl`.
2. Add a release workflow that builds and publishes `packages/verifiers-rl`.
3. Tag the first version and run the workflow.

Notes:

- The initial PyPI project association and ownership still require one-time account/project setup in PyPI.
- After that, publishing can be fully automated via CI.

## Why this is better than alternatives

### Alternative A: Keep in `verifiers/rl/` and rely only on extras

- Still burdens local static analysis/indexing in monorepo development.
- Blurs ownership/maintenance status.
- Keeps RL as an apparent core subsystem.

### Alternative B: Move RL examples under `environments/`

- Wrong abstraction boundary (trainer is not an environment).
- Harder to preserve CLI workflow and packaging expectations.

### Alternative C: Separate external repo immediately

- Highest maintenance overhead right away.
- Loses convenient in-repo co-development.
- Harder migration path.

Recommended subpackage approach gives most of the decoupling without cross-repo overhead.

## Migration checklist

1. Create `packages/verifiers-rl` package with copied RL code and scripts.
2. Update docs (`docs/training.md`, `docs/faqs.md`) to install `verifiers-rl`.
3. Add compatibility shims + deprecation warnings in core.
4. Add release workflow for `verifiers-rl` publishing.
5. Release with migration notes.
6. Remove compatibility exports in a later minor release.
