# Runtimes

A `Runtime` provides command and file access inside a sandbox with immutable user, `HOME`, and environment bindings. Task and harness hooks receive it as `runtime: vf.Runtime`, while configuration fields such as `agent.runtime` (and `--env.agent.runtime.type`, which defaults to `prime`) describe the runtime policy used to provision the sandbox. Because Python task hooks like `setup`, `finalize`, and `@vf.reward` always execute in the evaluator process, use their `runtime` parameter whenever you want to read files or run commands inside the sandbox; calling `Path(...).read_text()` directly reads from the evaluator host instead.

## Capabilities

Every runtime runs commands, reads and writes files, and starts background services. The runtimes differ in where they run and which features they support:

- **`subprocess`** runs commands directly on the evaluator host, with full access to local files, processes, and networking. Use it for fast local debugging with lightweight harnesses such as `bash` or `null`.
- **`docker`** runs each rollout in a local container. Use it for local testing of tasks that need an image or a container-only harness. CPU, memory, and GPU limits depend on the host; GPU pass-through requires the NVIDIA Container Toolkit.
- **`prime`** runs each rollout in a remote Prime sandbox, built for high-concurrency evaluation and RL training. It uses your Prime Intellect credentials, and it is the only runtime that enforces hard `disk` limits.
- **`custom`** loads a runtime from your own package. See [runtime plugins](#runtime-plugins).

Each runtime declares its features through its backend's class attributes, and network policy support through its config. Before it provisions anything, the framework refuses a run that needs a feature the runtime does not declare. The error names the runtime and the missing feature.

| Feature | Declared by | Built-in runtimes |
| --- | --- | --- |
| Task image, container-only harnesses, isolated scoring | `SUPPORTS_TASK_IMAGES` | `docker`, `prime` |
| Task network policy | config inherits `NetworkPolicyConfig` | `docker`, `prime` |
| Task `agent_user` | `SUPPORTS_USER_SWITCHING` | `docker`, `prime` |
| Tool server in its own runtime | `SUPPORTS_PORT_EXPOSURE` | `subprocess`, `docker`, `prime` |
| Compose services | `SUPPORTS_COMPOSE` | `docker`, `prime` |

Runtime capabilities are orthogonal to agent harness capabilities like MCP tool support, image inputs, or conversation resumption; see [harnesses](harnesses.md) for harness features and [building environments](building-environments.md#separate-agent-and-grading-sandboxes) for running separate solver and grading sandboxes.

## Images, resources, and timeouts

Tasks can declare their own container image, working directory, hardware resources (CPU cores, memory in GB, and disk in GB), phase timeouts (in seconds), and network rules directly on `TaskData`:

```python
data = vf.TaskData(
    prompt="Write the result to /workspace/answer.txt.",
    image="registry.example.com/my-task:1",
    workdir="/workspace",
    resources=vf.TaskResources(cpu=2, memory=4, disk=10),
    timeout=vf.TaskTimeout(setup=300, agent=1800, scoring=120),
    network=vf.NetworkPolicyConfig(allow=[]),
)
```

When resolving a sandbox configuration, `TaskData.image` takes precedence over `runtime.image` so individual tasks in a dataset can bring their own containers. By contrast, `TaskData.workdir` and `TaskData.resources` act as task-specific defaults: they apply when the runtime config leaves those fields at their default values, while explicit runtime overrides (such as `--env.agent.runtime.cpu 4`) take priority. On `prime`, setting `runtime.checkpoint` boots the sandbox from an existing filesystem checkpoint instead of a container image, replacing both `runtime.image` and `TaskData.image` and preserving the checkpoint's disk size while still applying your CPU and memory settings.

Timeouts and token budgets can be tuned at evaluation time under `env.agent`:

- **Phase timeouts**: set `setup`, `rollout`, `finalize`, or `scoring` under `env.agent.timeout` to override the corresponding `TaskData.timeout` values (any phase left unset falls back to the task's timeout, or runs without a limit if neither is set).
- **Turn and token budgets**: set `max_turns`, `max_input_tokens`, `max_output_tokens`, or `max_total_tokens` on `env.agent` to cap an entire agent rollout, and `sampling.max_tokens` to cap each individual model response. Rollout token budgets are checked between turns, so the turn that crosses a threshold completes before the rollout stops. `max_turn_tokens` on `env.agent` also caps each response (it lowers `sampling.max_tokens` to its value), and stops the rollout after a response that hits it.

## Execution users

On `docker` and `prime`, you can set `TaskData.agent_user` to run the agent under an existing non-root user account in the container image (with a writable home directory). Task `setup` still runs as the container's default user, after which harness setup, the agent process itself, and any colocated task tools execute as `agent_user`. Once the agent finishes, `finalize`, scoring, and artifact collection switch back to the default user.

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

## Runtime plugins

verifiers ships only the `prime` sandbox runtime. Run other providers, such as Modal or
E2B, as plugins. Select `type = "custom"` to use a `RuntimeBackend` subclass from your
own package:

```toml
[env.agent.runtime]
type = "custom"
import_path = "my_package.runtime.MyBackend"
region = "us-east-1"
```

The backend sets `config_class` to its own `CustomRuntimeConfig` subclass. That class
validates every field next to `import_path` when the config loads. The framework
constructs the backend as `MyBackend(config, name)` with the validated config, like
the built-in backends. A backend with `SUPPORTS_TASK_IMAGES` declares `image` and `workdir`
fields, and a config that inherits `NetworkPolicyConfig` enforces task network
policies:

```python
from typing import ClassVar

from verifiers.v1.runtimes import CustomRuntimeConfig, CustomRuntimeInfo, RuntimeBackend


class MyConfig(CustomRuntimeConfig):
    region: str
    image: str = "python:3.11-slim"
    workdir: str | None = None


class MyBackend(RuntimeBackend):
    config_class = MyConfig
    is_local: ClassVar[bool] = False
    SUPPORTS_TASK_IMAGES: ClassVar[bool] = True

    def __init__(self, config: MyConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = CustomRuntimeInfo(**config.model_dump())
```

A backend implements this contract; `Runtime` views add user and environment
bindings and the uv-script helpers on top of it:

| Method | Required | Purpose |
| --- | --- | --- |
| `start()` | yes | Provision the resource. |
| `cleanup()` or `teardown()` | yes | Free it. `cleanup` is synchronous and also runs at interpreter exit. |
| `exec(runtime, argv, env)` | yes | Start a process and return a handle that streams its stdio. |
| `write(runtime, path, data)` | yes | Write a file, creating parent directories. |
| `run(runtime, argv, env)` | no | Run to completion. The default drains an `exec`. |
| `run_background(runtime, argv, env, log)` | no | Start a long-lived process with its output in `log`. The default keeps an `exec` open for the process's life; override it when the provider runs detached jobs. |
| `read(runtime, path, max_bytes)` | no | Read a file. The default reads through `run`. |
| `expose(port)` | with `SUPPORTS_PORT_EXPOSURE` | Publish a port to a URL outside the runtime. |
| `prepare_execution(routes, network)` | with `NetworkPolicyConfig` | Apply the network policy, or `network` in its place, before the agent starts; `None` routes reopen the network. |
| `host_url(url)` | no | Rewrite a host URL for programs inside the runtime. |

Install the plugin's package in every process that loads the config, because
validation imports the backend. The capability flags and `is_local` default to off.
`tests/unit/test_runtimes.py` has a complete example.

### Checking a runtime

`tests/unit/test_runtimes.py` tests the backend contract and every runtime feature. A
feature the runtime declares must work, and a feature it does not declare must be
refused before anything is provisioned. Pass your plugin's config with `--runtime` to
run the tests against it. Run them before you use the plugin, and after you change it:

```bash
uv run pytest tests/unit/test_runtimes.py --runtime \
    '{"type": "custom", "import_path": "my_package.runtime.MyBackend", "region": "us-east-1"}'
```

`--runtime` also takes a built-in type, for example `--runtime docker`. The tests
expect `sh` and Python 3 in the runtime's default image.

## Deployment and services

A `Deployment` coordinates all execution resources for a single rollout attempt: provisioning, placement, connectivity, and cleanup of the harness environment, the task workspace, supporting sidecar services, and fresh verification targets.

The **workspace** (`deployment.workspace`) is the `Runtime` where task commands run and task files live; setting `agent.deployment.mode = "colocated"` places the harness and colocated tools inside that same workspace.

When a task defines **Compose services** by returning a `ComposeSpec` from `deployment_spec()`, the deployment brings up the multi-container stack and exposes the designated main service as `deployment.workspace` (for example, running `app` as the workspace while a `postgres` sidecar runs alongside it). Without a `ComposeSpec`, the deployment manages a single runtime.

Task hooks can request `deployment` by parameter name and call `deployment.target(name)` to get a `Runtime` view for a specific named service. A task tool server can also run inside a named Compose service (`ToolsetConfig.service`, on a Prime VM); the harness reaches it over the Compose network by the service's name, so the tool server can front internal services that the workspace cannot reach directly. `finalize_services()` collects sidecar artifacts before isolated verification.

Compose runs on local Docker or on a Prime VM. On Prime, one VM hosts the Docker daemon and every service, and its network policy applies to all of them. On local Docker, the evaluator's own Docker daemon runs the task's Compose file, and that file can mount host paths and request Docker privileges. So local Docker refuses Compose tasks unless you set `agent.deployment.trust_compose = true` (`--env.agent.deployment.trust-compose`). Only set it for task packages you trust. Local Compose does not enforce network policies, so run restricted Compose tasks on Prime.

When isolated scoring inherits a workspace image built inside the Compose VM, the deployment releases the solver VM and rebuilds the trusted task definition on a fresh VM for grading. Only the declared submission artifacts are restored there. A separately configured verifier image uses the normal image-provisioning path instead.

A tool server can also run on a Prime VM of its own, under an egress policy of its own: set its `ToolsetConfig.runtime` to a Prime runtime, such as `vf.PrimeConfig(image="example/tools:1", allow=["*"])` for a server that calls public APIs beside an offline workspace, or `allow=[]` for one that runs agent-submitted code with no egress at all. Beside a Prime workspace without Compose services, the server's VM boots with the workspace and joins it over a LAN network of the rollout's own, on which the workspace's VM may open connections to the server and nothing may open one to the workspace; the harness reaches the server at its VM's LAN name (`<sandbox-id>.sandbox.internal`), and `deployment.target(name)` gives task hooks the server's VM. The VM's policy applies when the workspace's does, after setup. Elsewhere such a server keeps its runtime's published port, which takes no network policy. On Prime, the published port is a public HTTPS URL (`https://<port>-<sandbox-id>.sandbox-v2-prod.pinfra.io`) without authentication.

To validate an isolated task without calling a model, finalize the submission, collect its artifacts, and call `vf.grade_submission(task, trace, runtime)`. This runs the task's normal grader, verifier image, timeouts, and retries in a fresh verification target without taking ownership of the supplied solver runtime. As in a rollout, the verifier derives from the agent's runtime config on the trace (`trace.agent.config.runtime`), not from the solver's resolved config, so the solver task's requirements, such as its network policy, don't carry over. Inside `Task.validate`, request `trace` by name for vf-validate's trace, which records its runtime config.

## Network policies

A task's policy (`TaskData.network`, a `vf.NetworkPolicyConfig` with `allow` and `block`) combines with the runtime's own rules (`runtime.allow` and `runtime.block`) to control what the sandbox can reach once the agent starts:

- `["*"]` adds no restriction of its own, deferring to whichever policy is set on the other side.
- `[]` blocks external egress and permits only connections required by verifiers (such as model API interception and MCP tool traffic).
- When both the task and the runtime specify allowlists, a destination must match both lists; when both specify blocklists, the blocked destinations are merged. (A specific destination allowlist cannot be combined with a blocklist.)

For example, to give the agent general internet access while blocking Hugging Face domains:

```toml
[env.agent.runtime]
type = "prime"
allow = ["*"]
block = ["huggingface.co", "*.huggingface.co"]
```

Network policies take effect after task and harness `setup` complete (so `setup` can still download packages or assets) and remain active through `finalize` and scoring (except in Harbor's shared verifier mode, which [reopens network access during grading](harbor.md#network-policies)).

Scoring can have a policy of its own. `ScoringConfig.network` is the task's policy while scoring runs, in place of `TaskData.network`, and combines with the runtime's rules the same way. Verifiers applies it right before the scoring hooks run, in the workspace (`shared`) or the fresh verifier (`isolated`), so a grader that installs packages under a no-egress task needs no network code:

```python
vf.TaskData(
    ...,
    network=vf.NetworkPolicyConfig(allow=[]),
    scoring=vf.ScoringConfig(
        network=vf.NetworkPolicyConfig(allow=["pypi.org", "files.pythonhosted.org"])
    ),
)
```

Unset, isolated scoring runs under `TaskData.network` and shared scoring keeps the agent phase's network. Shared scoring can't restrict a runtime that started without a policy; use isolated scoring for that. A hook that needs a different policy at another point can call `runtime.set_network(policy)`, which enforces the policy as given on the whole runtime.

These egress rules govern processes inside the sandbox (including task tools configured with `colocated = true`), not Python hooks running on the evaluator or tools running in separate containers. For provider-hosted server tools that execute outside the sandbox, verifiers forwards supported domain restrictions to web search tools and disables any provider tools it cannot constrain (see [Harbor network policies](harbor.md#network-policies)). By default, restricted runs also append a short notice to the agent's system prompt (or first user message when the harness cannot append a system prompt) explaining the network policy; pass `--no-env.agent.network-notice` if a benchmark requires the unmodified prompt verbatim.
