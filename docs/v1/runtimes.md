# Runtimes

A runtime is where the agent program and task commands run. Select it with
`--env.agent.runtime.type`; the default is `prime`.

Python hooks such as task `setup` run on the evaluator. Use their `runtime`
parameter to read files or run commands in the sandbox. A plain
`Path(...).read_text()` reads a file on the evaluator's machine.

## Capability matrix

These are the features exposed by the Verifiers runtime adapters. All runtimes
support commands, file reads and writes, background services, and live processes.

| Capability | `subprocess` | `docker` | `podman` | `apptainer` | `prime` | `modal` |
| --- | --- | --- | --- | --- | --- | --- |
| Runs on | Evaluator host | Local container engine | Local container engine | Apptainer host | Remote sandbox | Remote sandbox |
| Task image and container-required tasks | No | Yes | Yes | Yes | Yes | Yes |
| `cpu` / `memory` settings | No | Yes | Yes | Yes | Yes | Yes |
| Enforced `disk` size | No | No | No | No | Yes | No |
| GPU selection | No setting | Count; checks requested type | NVIDIA CDI device count | All accessible NVIDIA GPUs | Type and count | Type and count; no GPUs with `vm = true` |
| Execution-time allowlist | No | Yes | Yes | No | Hosts | HTTPS domains on port 443 |
| Execution-time deny list | No | Yes | Yes | No | Hosts | No |
| Framework-only networking (`allow = []`) | No | Yes | Yes | No | Yes | Yes |
| Artifact transfer and isolated grading | No | Yes | Yes | Yes | Yes | Yes |
| Harbor Docker Compose | No | Yes, with conditions below | No | No | Yes, in a VM | Yes, in an experimental VM |

Local CPU, memory, and GPU settings require support from the host and container
engine. Docker GPU selection needs the NVIDIA container toolkit; Podman needs
NVIDIA CDI devices. Apptainer passes CPU and memory flags to its CLI, and `gpu`
enables `--nv`; its GPU count and type are not enforced. Local containers and
Modal accept `disk` as an advisory request.

`subprocess` can access the evaluator's files, processes, devices, and network.
Apptainer also shares the host network. Tasks that require network restrictions
are rejected on these two runtimes. Use `subprocess` for trusted local development
with a compatible harness, such as `bash` or `null`.

Modal requires the `modal` extra and Modal credentials. Its ordinary sandboxes
support GPU requests; its VM backend is CPU-only. Prime uses Prime credentials.
GPU availability depends on the provider or host.

[Harbor Compose](harbor.md#docker-compose) supports CPU tasks on Docker, Prime,
and Modal. Local Docker requires unrestricted networking and `--env.trust-compose`.
Modal requires access to its experimental VM backend and `network_access = true`.
Ordinary runtime networking support does not imply the same support for Compose.

Harness capabilities such as MCP, image input, and conversation resume are
separate; see [harnesses](harnesses.md). For separate agent and grading sandboxes,
see [building environments](building-environments.md#separate-agent-and-grading-sandboxes).

Set `Task.NEEDS_CONTAINER = True` when a task needs its own filesystem or runs
untrusted code. Setting `TaskData.image` also requires a container. Most
third-party harnesses require one too; they cannot use `subprocess`.

## Images, resources, and timeouts

Put each task's requirements in `TaskData`:

```python
data = vf.TaskData(
    prompt="Write the result to /workspace/answer.txt.",
    image="registry.example.com/my-task:1",
    workdir="/workspace",
    resources=vf.TaskResources(cpu=2, memory=4, disk=10),
    timeout=vf.TaskTimeout(setup=300, agent=1800, scoring=120),
    network_allow=[],
)
```

The task's image overrides `runtime.image`. The task's workdir and resources are
used when the corresponding runtime settings are still at their defaults.
Non-default runtime settings win for those fields.

CPU is in cores, memory and disk in GB, and timeouts in seconds. Prime enforces
disk limits; local containers and Modal treat them as requests only.

Limits under `env.agent` apply to each agent rollout: `max_turns`,
`max_input_tokens`, `max_output_tokens`, and `max_total_tokens`.
`sampling.max_tokens` limits each model response. Token limits are checked
between turns, so the turn that exceeds a limit can finish.

Under `env.agent.timeout`, set `setup`, `rollout`, `finalize`, or `scoring` to
override the task's timeouts. Unset values use the task's timeout, or no limit
if the task has none. Explicit timeouts must be greater than zero.

## Runtime operations

Use `runtime` in task hooks and scoring:

```python
async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
    await runtime.write("input.txt", b"task input\n")
    result = await runtime.run(["sh", "-c", "mkdir -p results"], {})
    if result.exit_code:
        raise RuntimeError(result.stderr)


async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
    trace.info["answer"] = (
        await runtime.read("results/answer.txt", max_bytes=1_000_000)
    ).decode()
```

- `run(argv, env)` returns the exit code, stdout, and stderr in a `ProgramResult`. Check the exit code; a failed command does not raise automatically.
- `read` returns bytes; `write` takes bytes.
- `run_uv_script(script, args, env)` runs Python source with inline dependency metadata. Install dependencies during setup if the agent will run offline.
- `run_background(argv, env, log)` starts a service. Check that it is ready before using it.

Ordinary container runtimes replace the image entrypoint with a process that
keeps the container alive. Start required services yourself. Harbor
[Compose](harbor.md#docker-compose) keeps service entrypoints.

Use `Task.runtime_env()` to pass environment variables without saving them in
task data. For harness credentials, use `harness.forward_env = ["TOKEN_NAME"]`
so the config stores the variable's name, not its secret value. Programs in the
runtime can read these variables. Keep judge-only credentials on the evaluator.

## Network policies

Task `network_allow` / `network_block` combine with runtime `allow` / `block`:

- `["*"]` adds no restriction; the other policy still applies.
- `[]` allows only connections needed by verifiers, including model requests and MCP.
- If both sides list allowed destinations, a destination must be allowed by both.
- Blocklists combine. A nonempty allowlist of specific destinations cannot be used with a blocklist; validation rejects that combination.

For example, to allow the web except a dataset host:

```toml
[env.agent.runtime]
type = "docker"
allow = ["*"]
block = ["huggingface.co", "*.huggingface.co"]
```

Task and harness setup runs before restrictions start. The restrictions then
remain active through finalization and scoring. They apply inside the runtime,
not to Python hooks on the evaluator or tools in a separate runtime. Set
`colocated = true` for a task tool to share the agent's files and network rules.

Provider-hosted tools run outside the sandbox. Verifiers passes supported
restrictions to web tools or disables tools it cannot restrict; see
[Harbor network policies](harbor.md#network-policies).
On Modal, use `network_access = true` with an allowlist to keep setup online.
`network_access = false` blocks setup too.
