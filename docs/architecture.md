# Architecture

verifiers is built out of the following parts:

A server-backed evaluation or prime-rl **orchestrator** creates worker processes and distributes rollout requests among them. The client owns the taskset. It loads the tasks once and ships each task to the workers, which owns the runtime containing the agent(s) to produce a trace out of the data.

The orchestrator and workers are managed by verifiers and prime-rl themselves and thus offer few configurable knobs.

The **rollout** is the executable combination of one loaded task, the harness, and any tools. Each rollout has an independent trace and runtime state. verifiers has several runtimes which you can use for most tasksets:

- The `subprocess` runtime runs the rollouts in Python subprocesses locally. Thus, it is meant for debugging purposes, as there might be side effects during runtime, such as one subprocess altering the config files of the harness, which then affects the other subprocesses.
- The `docker` runtime runs the rollouts in docker containers on your local machine.
- The `podman` runtime is the same implementation driving the Podman CLI, for hosts without Docker.
- The `apptainer` runtime runs the rollouts in unprivileged Apptainer instances on the host network, as on HPC clusters. It has no egress policy.
- Sandbox runtimes, such as `prime`, `modal`, `e2b`, or `vercel`, are meant for production, especially for training or higher concurrency evaluation. These runtimes run remotely.

For offline Docker/Podman use on Linux, cache the task image and, if it lacks Python 3, `docker.io/library/python:3.11-alpine` for host callbacks. Restricted execution also needs the cached `localhost/verifiers-network:1` image, built during the first online startup.

The harness runs inside the rollout runtime to interact with the taskset. The harness does _not_ call the provider endpoint directly. Instead, model traffic goes through an **interception server** over a local connection or [Prime Tunnel](https://docs.primeintellect.ai/sandboxes/tunnel).

The interception server receives all these requests and then sends them over to the actual API, e.g. the OpenAI responses endpoint. It uses the endpoint that the harness expects, so Codex will use OpenAI Responses, while Claude Code will use the Anthropic Messages API.

The interception server, however, allows several things beyond just replaying the correct API response:

- Traces are built live, thus allowing the collection of the trajectories as they happen
- Setting sampling parameters in harnesses that don't necessarily expose those settings
- Intercepting and rewriting tool responses or server-side web search results to block reward hacks

## Finishing a rollout

When the agent ends its turn, the rollout collects and grades the agent's environment before it closes the harness session:

1. `task.finalize` runs and the artifacts are collected, including the main service's `[[verifier.collect]]` commands.
2. When grading runs in the agent's own runtime, the task scores it: `task.score`, then `on_score`.
3. The harness session closes, then the tool servers and the interception server.
4. Harness metrics (`harness.score`) run.

With isolated grading, the session closes right after step 1. The harness metrics run next, then the workspace stops, sidecar evidence is collected (`finalize_services`), and the task is graded in a fresh runtime.

Harnesses stop the processes their commands started when their session closes. For example, nano-rlm kills each job's process group, and codex and prime-agent terminate their shells. Anything the agent leaves running, such as a server or daemon a task asks it to keep up, therefore exists only while the session is open. Grading in the agent's runtime and collect commands read that live state, so they run first. Harness metrics run after the close because the close hands over the session's final state.

A rollout stopped mid-turn, by an agent timeout or a stop condition, closes the session first, so an agent that may still be acting cannot change the runtime while it is graded. A failed rollout is not scored.
