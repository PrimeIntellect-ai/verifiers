## `verifiers.rl` compatibility shim

The RL trainer implementation has moved to the optional `verifiers-rl` package.

Install:

```bash
uv add verifiers-rl
```

New source location:

- `packages/verifiers-rl/verifiers_rl/rl/`

Compatibility imports under `verifiers.rl` are retained temporarily to ease migration.
