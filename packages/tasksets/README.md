# tasksets

Optional taskset package for `verifiers`.

Install:

```bash
uv add tasksets
```

This package provides reusable task collections for `ComposableEnv`:

- `tasksets.base` for `Task`, `TaskSet`, `SandboxTaskSet`, `SandboxSpec`, and
  `TaskRuntimeSpec`
- `tasksets.harbor` for Harbor-format task directories
- `tasksets.swe_bench` for SWE-bench Pro / SWE-bench style tasksets
- `tasksets.terminal_bench` for Terminal-Bench 2 tasksets

The package keeps task data and task-specific grading out of individual
environment modules. Environments compose these tasksets with agent harnesses
from the separate `harnesses` package.

`SandboxSpec` describes sandbox creation fields such as image, start command,
resources, timeout, and sandbox environment variables. `TaskRuntimeSpec` wraps
that per-task sandbox spec together with workdir, task env vars, timeouts,
task-provided tools, task-provided skills, and upload directories so composable
environments can read one runtime object instead of many parallel getters.
