# Agent

An `Agent` is a configured `harness` with a model running in a `Runtime`. It can be configured with an `AgentConfig`. An agent is given a `Task` and produces a `Trace`.

```python
async with vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2")) as solver:
    trace = await solver.run(vf.Task(vf.TaskData(prompt="What is 2+2?")))
```

`agent.interaction(task)` holds a rollout open turn by turn. The caller acts as the
user, and each `turn()` runs one harness segment (i.e., a message, tool call, tool result etc.). A `Segment` therefore contains messages, tool calls etc.

```python
async with agent.interaction(task) as interaction:
    segment = await interaction.turn("hello")
    if not segment.terminated:
        segment = await interaction.turn(f"you said: {segment.last_reply}")

trace = interaction.trace
```

See [Runtime](runtime.md) for execution handles, resource ownership, and deployment
of the harness, task workspace, and supporting services.

## Separate task execution

`runtime` selects the provider. `deployment.mode` has two built-in choices:

- `colocated` (default): the harness and tools run in the same workspace, on any runtime.
- `split`: the harness runs directly in a Prime VM and tools run in a Docker
  container inside that VM. Other providers reject this mode.

A workspace is the execution environment containing the task’s files and processes,
accessed through a `Runtime`. It may be a standalone container or VM, or the
selected service in a Compose group. It is not just the working directory.

Compose defines the task's services independently of this choice. In colocated
mode, the harness runs in the workspace service. In split mode, it runs outside
all task containers; tools run in the workspace service and can reach sidecars
by service name. Compose requires Docker or a provider VM with Docker support.

For a harness in a Prime VM with a nested Docker workspace:

```toml
[env.agent.harness]
id = "rlm"
builtin_tools = ["ipython"]
builtin_skills = ["bash", "edit"]

[env.agent.deployment]
mode = "split"

[env.agent.runtime]
type = "prime"
image = "python:3.11-slim"
cpu = 2
memory = 8
disk = 20

[env.agent.execution]
type = "docker"
image = "python:3.11-slim"
memory = 4
allow = []
```

`execution` optionally overrides the split container settings; omitting it uses
the defaults. Task image, workdir, resource requests, and network rules resolve
against this container configuration. Task hooks and artifacts use that runtime. Harness setup, model
access, and harness metrics use the harness `runtime`. With nested placement,
Docker is installed on demand
on Debian-based harness images; other images must provide a running Docker daemon.
Task memory defaults to 75% of the VM allocation and must leave space for the harness.
For Compose, that is a total service budget; services without explicit limits share
it equally. CPU defaults to 75% of the allocation. Each service has a process-count
limit and bounded container logs.

With nested Prime placement, a private Docker daemon stores execution images, writable layers, and
volumes on a reserved filesystem. `execution.disk` sets its total size in GiB,
including images; it defaults to 60% of the outer disk allocation. Provisioning
fails if it cannot reserve the space while leaving at least 1 GiB for the harness.

The RLM split adapter supports native IPython, bash, edit, and fetch tools and
builtin skills. Supervisor shell jobs, path watchers, uploaded skills,
colocated task tools, borrowed runtimes, GPU limits, and selective task egress
rules are unsupported. Task networking can be unrestricted or disabled with an
empty allowlist. The harness keeps its independently configured network access.
With nested placement, the task container is restarted after execution to stop remaining agent processes
before finalization and scoring, retaining its filesystem. Use this mode for tasks
whose workspace submissions depend on filesystem state. Compose sidecars stay alive
for evidence collection and grading. With an empty allowlist, split Compose keeps
service DNS and internal connections while removing external egress.

Harnesses must explicitly support split execution. Unsupported harnesses and
provider combinations fail before provisioning. Harnesses receive the workspace
handle and a worker launch connection through `setup_execution`.

`agent.scoring.mode` selects `shared` or `isolated` scoring. Tasks declare defaults
through `TaskData.scoring`, using the same `ScoringConfig`. Explicit agent fields
override task fields; otherwise scoring defaults to `shared`. Isolated verification
restores collected artifacts into a fresh runtime before task scoring.
`agent.scoring.runtime` overrides the isolated verifier's runtime.
`agent.scoring.env` merges over task scoring variables in either placement;
`${VAR}` and `${VAR:-default}` templates resolve only when scoring starts.
`agent.scoring.retries` uses `RetryConfig` for isolated scoring, with a default
`max_retries = 2`. Its rules select retryable errors; retries reuse the completed
submission in a fresh scoring environment. Shared scoring is not retried.
