# Architecture

verifiers is built out of the following parts:

A server-backed evaluation or prime-rl **orchestrator** creates worker processes and distributes rollout requests among them. The client owns the taskset: it loads the tasks once and ships each task to the workers, which own the runtime containing the agent(s), score the result, and return the episode and its traces.

The orchestrator and workers are managed by verifiers and prime-rl themselves and thus offer few configurable knobs.

The **rollout** is the executable combination of one loaded task, the harness, and any tools. Each rollout has an independent trace and runtime state. verifiers supports several [runtimes](runtimes.md) that you can use for most tasksets:

- The `subprocess` runtime runs rollouts in local Python subprocesses. It is meant for debugging, since processes share your machine and can have side effects on one another (such as altering a harness's config files).
- The `docker` runtime runs rollouts in Docker containers on your local machine.
- The `podman` runtime uses the same container implementation through the Podman CLI for hosts without Docker.
- The `apptainer` runtime runs rollouts in unprivileged Apptainer instances on the host network, as on HPC clusters, without network egress filtering.
- Sandbox runtimes, such as `prime`, `modal`, or `e2b`, run remotely and are meant for production workloads like training or high-concurrency evaluation.

For offline Docker/Podman use on Linux, cache the task image first and, if it lacks Python 3, `docker.io/library/python:3.11-alpine` for host callbacks. Network-restricted execution also needs the cached `localhost/verifiers-network:1` image, which is built during the first online startup.

The harness runs inside the rollout runtime to interact with the task. Rather than calling the model provider directly, the harness sends model traffic through verifiers' **interception server** over a local connection or [Prime Tunnel](https://docs.primeintellect.ai/sandboxes/tunnel).

The interception server receives these requests and forwards them to the upstream provider using the API surface that the harness expects, such as OpenAI Responses for Codex or Anthropic Messages for Claude Code.

Routing traffic through the interception server enables several things beyond forwarding the provider response:

- Building traces live so trajectories and token usage are recorded as they happen
- Applying sampling parameters in harnesses that don't expose those settings directly
- Running the task's [stop and interception hooks](tasksets.md#stops-and-interception) to inspect or rewrite tool responses and server-side web search results (for example, to block reward hacks)

## Finishing a rollout

When the agent finishes its turn normally, the rollout collects artifacts and runs in-place grading before closing the harness session:

1. `task.finalize` runs and declared artifacts are collected, including the main service's `[[verifier.collect]]` commands.
2. When grading runs in the agent's own runtime (`shared` scoring), the task scores the workspace (`task.score`, followed by `on_score`).
3. The harness session closes, followed by the tool servers and the interception server.
4. Harness metrics (`harness.score`) run.

With isolated grading (`isolated` scoring), the harness session closes right after step 1, harness metrics run next, the workspace stops, sidecar evidence is collected (`finalize_services`), and the task is graded in a fresh runtime.

Closing the harness session terminates any processes started by the agent's commands (for example, `nano-rlm` kills each job's process group, and `codex` and `prime-agent` close their shells). Running artifact collection and shared grading before closing the session ensures they can inspect live background processes or servers that the task asked the agent to leave running, while harness metrics wait until session close hands over the final session state.

If a rollout is stopped mid-turn by a timeout or a stop condition, the harness session closes first so an still-active agent cannot mutate the workspace while grading runs. Rollouts that fail with an unhandled error are not scored.
