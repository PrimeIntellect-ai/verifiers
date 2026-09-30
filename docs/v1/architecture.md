# Architecture

A run follows these steps:

1. The client loads tasks from the taskset and sends them to workers.
2. Each worker starts the runtime and harness for an agent run.
3. The harness sends model requests through verifiers, which records the conversation.
4. The task scores the result, and verifiers saves the episode and its traces.

verifiers manages the workers for evaluations; prime-rl manages them for training.
Tasks are loaded once on the client, rather than separately in every worker.

## Where programs run

Each agent runs in a runtime:

| Runtime | Where it runs |
| --- | --- |
| `subprocess` | Processes on your machine; files and settings can affect other runs |
| `docker` / `podman` | Local containers |
| `apptainer` | Unprivileged containers, often on a cluster; shares the host network |
| `prime` / `modal` | Remote sandboxes, suited to many concurrent runs |

See [runtimes](runtimes.md) for images, resources, files, and network restrictions.

For offline Docker/Podman on Linux, cache the task image first. If it lacks
Python 3, also cache `docker.io/library/python:3.11-alpine` for calls back to the
host. Network restrictions need `localhost/verifiers-network:1`, which is built
during the first online startup.

## How model calls are recorded

The harness sends model requests to verifiers' **interception server**, which
forwards them to the model provider. The connection uses a local address or
[Prime Tunnel](https://docs.primeintellect.ai/sandboxes/tunnel).

The server accepts the API the harness expects: for example, OpenAI Responses
for Codex or Anthropic Messages for Claude Code. It records requests and
responses as they happen, applies sampling settings, and runs the task's stop
and interception hooks. Those hooks can inspect or change supported messages,
such as tool responses; see [tasksets](tasksets.md#stops-and-interception).
