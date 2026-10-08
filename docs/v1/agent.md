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

## Separate task execution

`runtime` selects the harness backend and optional `execution` selects the task
workspace backend. `deployment.strategy` controls their placement:

- `auto` uses nested containers for Prime/Docker plus Docker execution, independent
  allocations for other split pairs, and a shared target when execution is omitted.
- `nested` keeps Docker task containers inside a Prime VM, or runs the trusted
  harness on the evaluator with local Docker task containers.
- `independent` provisions the harness and execution separately. Their resource
  budgets are independent; the workspace need not fit inside the harness allocation.

For a harness in a Prime VM with a nested Docker workspace:

```toml
[env.agent.harness]
id = "rlm"
builtin_tools = ["ipython"]
builtin_skills = ["bash", "edit"]

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

Task image, workdir, resource requests, and network rules resolve against
`execution`. Task hooks and artifacts use that runtime. Harness setup, model
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
Local Docker rejects explicit execution disk quotas; it does not provide this
storage-containment guarantee.

The RLM split adapter supports RLM's IPython builtin and builtin skills. Native
bash/edit/fetch tools, supervisor shell jobs, path watchers, uploaded skills,
colocated task tools, borrowed runtimes, GPU limits, and selective task egress
rules are unsupported. Task networking can be unrestricted or disabled with an
empty allowlist. The harness keeps its independently configured network access.
With nested placement, the task container is restarted after execution to stop remaining agent processes
before finalization and scoring, retaining its filesystem. Use this mode for tasks
whose workspace submissions depend on filesystem state. Compose sidecars stay alive
for evidence collection and grading. With an empty allowlist, split Compose keeps
service DNS and internal connections while removing external egress.

Deployment strategies implement placement, connections, network phases, quiescing,
and verifier placement. `Runtime.deployment_strategy()` selects a strategy;
`register_deployment_strategy(name, factory)` registers an application strategy.
Each attempt gets a fresh strategy instance. A strategy binds `deployment.harness`
and `deployment.execution`, uses `allocate(name, config)` for additional targets,
and `enter(context_manager)` to own resources such as LAN membership. Cleanup is
registered before runtime startup, including partial failures.

`deployment.connection(source, target)` returns an `ExecutionConnection`. Its
`command(argv)` launches a worker from the source into the target; harnesses receive
it in `setup_execution(runtime, task_runtime, connection)`. The built-in transport
supports local/container exec. Independently placed remote VMs require a strategy
that supplies authenticated transport, readiness, network policy, and quiescing;
selecting two remote runtime configs does not provide those capabilities by itself.
Prime LAN provisioning is not implemented.

Fresh verifiers are deployment-owned targets, supervised during staging and
scoring. Independent placement defaults the verifier to the execution backend;
nested placement defaults it to the outer provider backend. `agent.verifier.runtime`
overrides that choice. Task and Env programs do not select placement or transport.

Execution-target loss can be scored as a terminal training sample:

```toml
[env.agent.execution_failure]
workspace_loss = "zero"
agent_timeout = "zero"
```

Workspace loss is confirmed through container or Prime lifecycle state; an unreachable Docker daemon
is not proof of loss. The trace records the cause, evidence, and selected policy
without treating unknown attribution as agent fault. A zero policy preserves a
valid trace with reward zero and skips unavailable workspace grading. Agent deadlines
include model latency; use this policy only when that is the intended task budget.
Provider/transport failures remain errors. Recovery is not implemented.
