# BYO Harness

Source: `docs/overview.md` and `docs/byo-harness.md`.

For composable environments with reusable tasksets, toolsets, custom programs,
or custom harnesses, Verifiers exposes the v1 BYO Harness path under
`verifiers.v1`.

A v1 environment combines a `vf.Taskset` with a `vf.Harness` through `vf.Env`.
The taskset owns the task collection and scoring signals. The harness owns the
rollout runner, model endpoint controls, tools, custom programs, and sandboxed
execution.
