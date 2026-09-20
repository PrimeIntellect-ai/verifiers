# Coordinated swarm smoke test

Four solvers independently check a joint result. One coordinator gathers five
identity-bound approvals of an exact repository commit and requests submission.
The controller performs public checks before recording acceptance. Agent model
requests stop through Verifiers' native stop hooks; the world remains available.

This tests coordination and submission, not code-generation quality. The initial
repository commit stays unchanged. It does not run FrontierSWE or hidden grading.

Install with `uv pip install -e environments/swarm_coordination`. Start a Worlds
server with the `decisions` service available, set `WORLDS_ADMIN_TOKEN` on the
controller, and provide its sandbox-reachable URL:

```bash
uv run eval swarm-coordination -n 1 -r 1 \
  --env.world.url http://127.0.0.1:8787 \
  --env.world.agent-url https://YOUR-WORLD-TUNNEL \
  --env.solver.max-turns 40 --env.coordinator.max-turns 50
```

Both roles use ordinary Verifiers agent configurations, including model, harness,
and runtime. Defaults are four solver copies, one coordinator, and concurrency
five. Each run provisions its own runtime. Never use subprocess placement for
untrusted coding agents.

Any participant can propose a revision. Each new proposal starts without votes;
only the coordinator can request acceptance, and only the review controller can
record the public-check result. Objections and withdrawals invalidate a pending
request. Changed main commits cannot be accepted under earlier approvals.

`review_timeout` bounds the coordination window. On expiry, the native stop hook
ends further model work and captures an unaccepted result; this smoke task scores
it zero. This is not yet a benchmark checkpoint-fallback implementation.
