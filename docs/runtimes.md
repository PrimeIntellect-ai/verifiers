# Runtimes

A `Runtime` provides command and file access inside a sandbox with immutable user, `HOME`, and environment bindings. Task and harness hooks receive it as `runtime: vf.Runtime`, while configuration fields such as `agent.runtime` (and `--env.agent.runtime.type`, which defaults to `prime`) describe the runtime policy used to provision the sandbox. Because Python task hooks like `setup`, `finalize`, and `@vf.reward` always execute in the evaluator process, use their `runtime` parameter whenever you want to read files or run commands inside the sandbox; calling `Path(...).read_text()` directly reads from the evaluator host instead.

## Capability matrix

Every runtime adapter in verifiers supports running commands, reading and writing files, starting background services, and managing live processes, but they differ in how they handle isolation, hardware resources, and networking:

| Capability | `subprocess` | `docker` | `podman` | `apptainer` | `prime` | `modal` | `e2b` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Runs on | Evaluator host | Local container engine | Local container engine | Apptainer host | Remote sandbox | Remote sandbox | Remote sandbox |
| Task image and container-required tasks | No | Yes | Yes | Yes | Yes | Yes | Public Debian-based image |
| `cpu` / `memory` settings | No | Yes | Yes | Yes | Yes | Yes | Yes |
| Enforced `disk` size | No | No | No | No | Yes | No | No |
| GPU selection | No setting | Count; checks requested type | NVIDIA CDI device count | All accessible NVIDIA GPUs | Type and count | Type and count; no GPUs with `vm = true` | No setting |
| Execution-time allowlist | No | Yes | Yes | No | Hosts | HTTPS domains on port 443 | Hosts, IPs, CIDRs |
| Execution-time deny list | No | Yes | Yes | No | Hosts | No | IPs and CIDRs |
| Framework-only networking (`allow = []`) | No | Yes | Yes | No | Yes | Yes | Yes |
| `TaskData.agent_user` | No | Yes | Yes | No | Yes | No | No |
| Artifact transfer and isolated grading | No | Yes | Yes | Yes | Yes | Yes | Yes |
| Harbor Docker Compose | No | Yes, with conditions below | No | No | Yes, in a VM | Yes, in an experimental VM | No |

- **Local runtimes (`subprocess`, `docker`, `podman`, `apptainer`)**: `subprocess` runs commands directly on the evaluator host with full access to local files, processes, and networking. It is useful for fast local debugging with lightweight harnesses like `bash` or `null`, but tasks that set `Task.NEEDS_CONTAINER = True`, specify a `TaskData.image`, enforce network policies, or use third-party CLI harnesses require a container runtime instead. For local containers, CPU, memory, and GPU limits depend on the host and container engine: Docker GPU pass-through requires the NVIDIA Container Toolkit, Podman requires NVIDIA CDI devices, and Apptainer passes `--nv` to expose host GPUs without enforcing a specific device count or model. Apptainer also shares the host network, so it cannot run tasks with egress restrictions.
- **Remote sandbox runtimes (`prime`, `modal`, `e2b`)**: Remote sandboxes are built for high-concurrency evaluation and RL training. `prime` uses your Prime Intellect credentials and is the only backend that enforces hard `disk` limits (on other container runtimes, `disk` is treated as an advisory request). `modal` and `e2b` require installing their respective optional extras (`modal` or `e2b`) and configuring provider credentials. On Modal, standard sandboxes support GPU requests, whereas its VM mode (`vm = true`) is CPU-only. On E2B, the adapter builds and caches a sandbox template from a public Debian-based image along with the requested CPU and memory settings; network allowlists accept wildcard hostnames, IPs, and CIDRs (without schemes or ports), while blocklists accept IPs and CIDRs only.
- **Multi-container Compose tasks and harness features**: Multi-container [Harbor](harbor.md) Docker Compose tasks are supported for CPU workloads on `docker` (with unrestricted networking and `--env.agent.deployment.trust-compose`), `prime` (inside a VM), and `modal` (using its experimental VM backend with `network_access = true`). Runtime capabilities are orthogonal to agent harness capabilities like MCP tool support, image inputs, or conversation resumption; see [harnesses](harnesses.md) for harness features and [building environments](building-environments.md#separate-agent-and-grading-sandboxes) for running separate solver and grading sandboxes.

## Images, resources, and timeouts

Tasks can declare their own container image, working directory, hardware resources (CPU cores, memory in GB, and disk in GB), phase timeouts (in seconds), and network rules directly on `TaskData`:

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

When resolving a sandbox configuration, `TaskData.image` takes precedence over `runtime.image` so individual tasks in a dataset can bring their own containers. By contrast, `TaskData.workdir` and `TaskData.resources` act as task-specific defaults: they apply when the runtime config leaves those fields at their default values, while explicit runtime overrides (such as `--env.agent.runtime.cpu 4`) take priority. On `prime`, setting `runtime.checkpoint` boots the sandbox from an existing filesystem checkpoint instead of a container image, replacing both `runtime.image` and `TaskData.image` and preserving the checkpoint's disk size while still applying your CPU and memory settings.

Timeouts and token budgets can be tuned at evaluation time under `env.agent`:

- **Phase timeouts**: set `setup`, `rollout`, `finalize`, or `scoring` under `env.agent.timeout` to override the corresponding `TaskData.timeout` values (any phase left unset falls back to the task's timeout, or runs without a limit if neither is set).
- **Turn and token budgets**: set `max_turns`, `max_input_tokens`, `max_output_tokens`, or `max_total_tokens` on `env.agent` to cap an entire agent rollout, and `sampling.max_tokens` to cap each individual model response. Rollout token budgets are checked between turns, so the turn that crosses a threshold completes before the rollout stops.

## Execution users

On `docker`, `podman`, and `prime`, you can set `TaskData.agent_user` to run the agent under an existing non-root user account in the container image (with a writable home directory). Task `setup` still runs as the container's default user, after which harness setup, the agent process itself, and any colocated task tools execute as `agent_user`. Once the agent finishes, `finalize`, scoring, and artifact collection switch back to the default user.

Calling `runtime.with_user(...)` or `runtime.with_env(...)` returns an independent view over the same underlying resource without mutating existing views. Under the hood, a `RuntimeBackend` (such as `DockerBackend` or `PrimeBackend`) owns the provider lifecycle, including container startup, network policy, and teardown; `backend.bind()` creates a `Runtime` view without provisioning a new resource, and `runtime.backend` exposes the backend when explicit infrastructure operations are needed. `Agent.provision()` lends a runtime whose cleanup remains owned by the caller's context.

Because setting `agent_user` does not create the Unix account or change existing file ownership, install system packages and create directories during image build or task `setup`, then grant `agent_user` permission to read its inputs and write to its working directory and output paths before `setup` returns. Note that a view is an execution handle rather than a security boundary: user permissions are enforced by the container OS, and if a task uses private tests or sensitive grading assets, you should stage them in a separate [grading sandbox](building-environments.md#separate-agent-and-grading-sandboxes).

## Runtime operations

Task lifecycle hooks (`setup`, `finalize`, `stage_verifier`) and `@vf.reward` / `@vf.metric` methods can access the active sandbox by declaring a `runtime: vf.Runtime` parameter:

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

The `vf.Runtime` interface provides a small set of methods for interacting with the sandbox:

- `run(argv, env)` executes a command and returns a `ProgramResult` containing `exit_code`, `stdout`, and `stderr`. Nonzero exit codes do not raise automatically, so check `result.exit_code` when a command must succeed.
- `read(path, max_bytes=...)` and `write(path, data)` transfer raw `bytes` between the evaluator and the sandbox filesystem.
- `prepare_uv_script(script)` and `run_uv_script(script, args, env)` prepare and execute self-contained Python scripts that declare PEP 723 inline dependencies.
- `run_background(argv, env, log)` launches a background service inside the sandbox. Because standard container runtimes replace the image entrypoint with a keep-alive process (unlike Docker Compose tasks, which preserve service entrypoints), use `run_background` in `setup` to start any daemons the image would normally launch and verify they are ready before returning.

To pass environment variables into sandbox commands without serializing them into `TaskData`, override `Task.runtime_env()` (or set `TaskData.scoring.env` / `agent.scoring.env` for variables used only during scoring). For credentials needed by an agent harness, set `harness.forward_env = ["TOKEN_NAME"]` so saved configs record only the variable name rather than the secret value. Processes inside the sandbox can read forwarded variables, so keep judge-only credentials on the evaluator host.

### Packaged grader scripts

Dependencies added to your environment package's `pyproject.toml` (via `uv add --project environments/<package> ...`) are installed on the evaluator for dataset loading and Python hooks, not inside the agent or grading containers. Libraries and CLI tools that the agent needs belong in the task image or `setup`, while dependencies for a sandboxed Python grader can either be baked into the verifier image or declared via PEP 723 inline script metadata.

If you bundle a self-contained grading script alongside your taskset and load its bytes with `importlib.resources`, you can pre-install its inline dependencies with `runtime.prepare_uv_script` and run it with `runtime.run_uv_script`:

```python
import json
from importlib.resources import files

GRADER = files("my_task").joinpath("check.py").read_bytes()


async def stage_verifier(self, runtime: vf.Runtime) -> None:
    await runtime.prepare_uv_script(GRADER)


@vf.reward
async def correct(self, runtime: vf.Runtime) -> float:
    result = await runtime.run_uv_script(GRADER, args=["/workspace/answer.py"])
    if result.exit_code:
        raise RuntimeError(result.stderr)
    return float(json.loads(result.stdout)["score"])
```

Calling `prepare_uv_script` during `stage_verifier` (or during `setup` for a public checker that runs inside the solver sandbox) installs the script's PEP 723 dependencies before execution-time network restrictions lock down the container. Passing the identical script bytes to `run_uv_script` during scoring then reuses the prepared interpreter offline. Whenever your grader imports or executes code written by the agent, run it inside a container (`NEEDS_CONTAINER = True`) rather than in a Python hook on the evaluator. Have the grader script exit `0` and emit a `0.0` score when the submission is wrong or crashes, and reserve nonzero exit codes (raising in `@vf.reward`) for failures that prevented a valid grade from being computed.

## Deployment and services

A `Deployment` coordinates all execution resources for a single rollout attempt: provisioning, placement, connectivity, and cleanup of the harness environment, the task workspace, supporting sidecar services, and fresh verification targets.

The **workspace** (`deployment.workspace`) is the `Runtime` where task commands run and task files live; setting `agent.deployment.mode = "colocated"` places the harness and colocated tools inside that same workspace.

When a task defines **Compose services** by returning a `ComposeSpec` from `deployment_spec()`, the deployment brings up the multi-container stack and exposes the designated main service as `deployment.workspace` (for example, running `app` as the workspace while a `postgres` sidecar runs alongside it). Without a `ComposeSpec`, the deployment manages a single runtime.

Task hooks can request `deployment` by parameter name and call `deployment.target(name)` to get a `Runtime` view for a specific named service. A task tool server can also run inside a named Compose service (`ToolsetConfig.service`, on a Prime or Modal VM); the harness reaches it over the Compose network by the service's name, so the tool server can front internal services that the workspace cannot reach directly. `finalize_services()` collects sidecar artifacts before isolated verification. On local Docker, Compose tasks require `agent.deployment.trust_compose = true` because task-authored Compose files can mount host paths and request Docker privileges.

To validate an isolated task without calling a model, finalize the submission, collect its artifacts, and call `vf.grade_submission(task, trace, runtime)`. This runs the task's normal grader, verifier image, timeouts, and retries in a fresh verification target without taking ownership of the supplied solver runtime.

## Network policies

Task-level rules (`TaskData.network_allow` and `TaskData.network_block`) combine with runtime-level rules (`runtime.allow` and `runtime.block`) to control what the sandbox can reach once the agent starts:

- `["*"]` adds no restriction of its own, deferring to whichever policy is set on the other side.
- `[]` blocks external egress and permits only connections required by verifiers (such as model API interception and MCP tool traffic).
- When both the task and the runtime specify allowlists, a destination must match both lists; when both specify blocklists, the blocked destinations are merged. (A specific destination allowlist cannot be combined with a blocklist.)

For example, to give the agent general internet access while blocking Hugging Face domains:

```toml
[env.agent.runtime]
type = "docker"
allow = ["*"]
block = ["huggingface.co", "*.huggingface.co"]
```

Network policies take effect after task and harness `setup` complete (so `setup` can still download packages or assets) and remain active through `finalize` and scoring (except in Harbor's shared verifier mode, which [reopens network access during grading](harbor.md#network-policies)). On Modal, keep `network_access = true` when using an allowlist so `setup` stays online; setting `network_access = false` cuts off networking during `setup` as well.

These egress rules govern processes inside the sandbox (including task tools configured with `colocated = true`), not Python hooks running on the evaluator or tools running in separate containers. For provider-hosted server tools that execute outside the sandbox, verifiers forwards supported domain restrictions to web search tools and disables any provider tools it cannot constrain (see [Harbor network policies](harbor.md#network-policies)). By default, restricted runs also append a short notice to the agent's system prompt (or first user message when the harness cannot append a system prompt) explaining the network policy; pass `--no-env.agent.network-notice` if a benchmark requires the unmodified prompt verbatim.
