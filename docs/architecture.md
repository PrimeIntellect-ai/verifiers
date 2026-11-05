# Architecture

verifiers is built out of the following parts:

An orchestrator creates worker processes and distributes rollout requests among them. The client owns the taskset: it loads the tasks once and ships each task to the workers, which own the runtime containing the agent(s), score the result, and return the episode and its traces.

The orchestrator and workers are managed by verifiers themselves and thus are less configurable.

A model in its harness solving a task produces a **rollout**. Each rollout has an independent trace and runtime state. verifiers supports several [runtimes](runtimes.md) that you can use for most tasksets:

- The `subprocess` runtime runs rollouts in local Python subprocesses. It is meant for debugging, since processes share your machine and can have side effects on one another (such as altering a harness's config files).
- The `docker` runtime runs rollouts in Docker containers on your local machine.
- The `prime` runtime runs rollouts in remote Prime sandboxes. It is meant for production workloads like training or high-concurrency evaluation.
- A `custom` runtime loads another provider from your own package, without upstreaming it. See [runtime plugins](runtimes.md#runtime-plugins).

The `subprocess` and `docker` runtimes are for quick local testing without billing, and support fewer features. `subprocess` refuses tasks that need an image, a network policy, or Compose services. Docker runs Compose services only for trusted tasks, with unrestricted networking. On Linux, an unrestricted Docker container shares the host network. On other platforms, Docker containers reach host services through `host.docker.internal`. A Docker container with a network policy runs on the bridge network, and its traffic goes through a verifiers proxy that enforces the policy.

The harness runs inside the rollout runtime to interact with the task. Rather than calling the model provider directly, the harness sends model traffic through verifiers' **interception server** over a local connection or [Prime Tunnel](https://docs.primeintellect.ai/sandboxes/tunnel).

The interception server receives these requests and forwards them to the upstream provider using the API surface that the harness expects, such as OpenAI Responses for Codex or Anthropic Messages for Claude Code.

Routing traffic through the interception server enables several things beyond forwarding the provider response:

- Building traces live so trajectories and token usage are recorded as they happen. This also prevents agents tampering with their own traces
- Applying sampling parameters in harnesses that don't expose those settings directly
- Running the task's [stop and interception hooks](tasksets.md#stops-and-interception) to inspect or rewrite tool responses and server-side web search results (for example, to block reward hacks)

### Relay

A Prime tunnel can drop for a while (its client reconnects) or for good (the tunnel service drops its registration). A lost tunnel fails every rollout on its interception server. With `[env.agent] relay = true`, a harness in a remote box reaches the interception server through a relay in its own box, at `127.0.0.1`, instead of the tunnel URL:

- While the tunnel cannot be reached, the relay retries the model call for up to 5 minutes, so the harness does not see a tunnel that reconnects.
- When a tunnel is lost for good, its interception server opens a new tunnel and points each relay at the new URL. A model call that was in flight goes through the new tunnel; the server joins it to the call it already serves, so the trace gets no second turn.

A rollout with tool servers or deployment services still fails on a lost tunnel, because they reach the interception server at the tunnel URL. The relay needs Python 3.10 or later in the box.

## Finishing a rollout

When the agent finishes its turn normally, the rollout collects artifacts and runs in-place grading before closing the harness session:

1. `task.finalize` runs and declared artifacts are collected, including the main service's `[[verifier.collect]]` commands.
2. When grading runs in the agent's own runtime (`shared` scoring), the task scores the workspace (`task.score`, followed by `on_score`).
3. The harness session closes, followed by the tool servers and the interception server.
4. Harness metrics (`harness.score`) run.

If a rollout is stopped mid-turn by a timeout or a stop condition, the harness session closes first so an still-active agent cannot mutate the workspace while grading runs. Rollouts that fail with an unhandled error are not scored.
