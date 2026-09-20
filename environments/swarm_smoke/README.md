# Swarm smoke test

Two agents exchange information through Worlds, independently verify a result,
and resolve a shared forge issue. Each uses its own runtime. The world remains
available in the local viewer after their episode accounts are deactivated.

Install this package with `uv pip install -e environments/swarm_smoke`.
Run a Worlds server separately, set `WORLDS_ADMIN_TOKEN` on the controller, and use:

```bash
uv run eval swarm-smoke --env.id swarm -n 1 -r 1 \
  --env.world.url http://127.0.0.1:8787 \
  --env.world.agent-url https://YOUR-SANDBOX-REACHABLE-WORLD \
  --env.agent.runtime.type prime --env.agent.max-turns 20
```

The controller credential is never passed into agent sandboxes. Each participant
receives only its own world credential. The example uses the Bash/Edit harness
and a small HTTP command-line client, with native Verifiers inference interception
and traces. Model and endpoint use the normal eval configuration.

`SwarmEnv` accepts copies per declared agent role through `participants`; additional
roles can be declared on a `SwarmEnvConfig` subclass with ordinary `AgentConfig`
fields. A `SwarmTask` defines world preparation, participant instructions, shared
submission capture, and team evaluation. This smoke test exercises communication
and forge state; it does not grade code or implement the FrontierSWE adapter.
