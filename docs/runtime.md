# Runtime

A `Runtime` provides command and file access with immutable user, `HOME`, and
environment bindings. Task and harness hooks receive it as `runtime: vf.Runtime`.
`runtime.with_user(...)` and `runtime.with_env(...)` return independent views of
the same resource; existing views keep their bindings. Working directory comes
from the runtime config.

An internal `RuntimeBackend` holds provider state and owns resource operations,
including startup, network policy, and teardown. `backend.bind()` creates an execution view without provisioning a resource.
Views share that backend and its caches. `runtime.backend` exposes it for explicit infrastructure operations;
ordinary task hooks only need the execution view. A view is not an isolation
boundary: backend permissions enforce the selected user.

`Agent.provision()` lends a runtime whose owning context controls cleanup.
Trusted task setup and grading use the default-user view; agent execution and
colocated tools use the task's agent-user view. No shared user or environment
state is mutated when deriving a view.

Use `runtime` (or a role-prefixed name such as `agent_runtime`) for execution
views, and `backend` (such as `harness_backend`) for lifecycle owners. Provider
implementations are named `DockerBackend`, `PrimeBackend`, and so on. Configuration
fields such as `agent.runtime` continue to describe runtime policy.

## Capabilities

Each runtime declares its features through its backend's class attributes, and
network policy support through its config. Before it provisions anything, the framework refuses a
run that needs a feature the runtime does not declare, and the error names the
runtime and the gap.

| Feature | Declared by | Built-in runtimes |
| --- | --- | --- |
| Task image, container-only harnesses, isolated scoring | `SUPPORTS_TASK_IMAGES` | docker, prime |
| Task network policy | config inherits `NetworkPolicyConfig` | prime |
| Task `agent_user` | `SUPPORTS_USER_SWITCHING` | docker, prime |
| Tool server in its own runtime | `SUPPORTS_PORT_EXPOSURE` | subprocess, docker |
| Nested containers, as Compose services need | `SUPPORTS_NESTED_CONTAINERS` | prime |

## Runtime plugins

verifiers ships only the `prime` sandbox runtime. Run other providers, such as Modal or
E2B, as plugins: select `type = "custom"` to use a `RuntimeBackend` subclass from your
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
bindings, `run_background`, and the uv-script helpers on top of it:

| Method | Required | Purpose |
| --- | --- | --- |
| `start()` | yes | Provision the resource. |
| `cleanup()` or `teardown()` | yes | Free it. `cleanup` is synchronous and also runs at interpreter exit. |
| `exec(runtime, argv, env)` | yes | Start a process and return a handle that streams its stdio. |
| `write(runtime, path, data)` | yes | Write a file, creating parent directories. |
| `run(runtime, argv, env)` | no | Run to completion. The default drains an `exec`. |
| `read(runtime, path, max_bytes)` | no | Read a file. The default reads through `run`. |
| `expose(port)` | with `SUPPORTS_PORT_EXPOSURE` | Publish a port to a URL outside the runtime. |
| `prepare_execution(routes)` | with `NetworkPolicyConfig` | Apply the network policy before the agent starts. |
| `host_url(url)` | no | Rewrite a host URL for programs inside the runtime. |

Install the plugin's package in every process that loads the config, because
validation imports the backend. The capability flags and `is_local` default to off.
`tests/unit/test_runtimes.py` has a complete example.

### Checking a runtime

`scripts/check_runtime.py` provisions a runtime and checks the backend contract and
every feature the runtime declares. Run it against your plugin before you use it, and
after you change it:

```bash
uv run scripts/check_runtime.py custom \
    --set import_path=my_package.runtime.MyBackend --set region=us-east-1
```

It prints `PASS`, `FAIL`, or `SKIP` (feature not declared) per check, and exits
non-zero on any failure. It also checks the built-in runtimes, for example
`uv run scripts/check_runtime.py docker`. The checks expect `sh` and Python 3 in the
runtime's default image.

## Deployment and services

A `Deployment` coordinates the execution resources for one rollout attempt:
provisioning, placement, connectivity, and cleanup of the harness environment,
task workspace, supporting services, and fresh verification targets. Backends
implement operations on each resource; the rollout controls the agent's execution.
Borrowed resources remain owned by the context that supplied them.

The **workspace** is the execution environment where task commands run and task
files live. `deployment.workspace` returns a `Runtime`; workspace names a
role, not a separate resource type or just a working directory.
`agent.deployment.mode = "colocated"` places the harness and tools in that workspace.

**Compose services** are the named containers in a task's Docker Compose file.
A task returns an optional `ComposeSpec` from `deployment_spec()` to describe this
setup and select its workspace service. For example, in a task with `app` and
`postgres` services, `app` can be the workspace while `postgres` runs alongside it.
Without a Compose specification, the deployment manages a single runtime.

Task hooks can receive `deployment` and use `deployment.target(name)` to access a
named service through a runtime view. `finalize_services()` collects sidecar evidence
before verification. See [Harbor's Compose support](harbor.md#docker-compose) for
provider details. Compose runs on Prime VMs.

For model-free validation of an isolated task, finalize the submission and collect
its artifacts, then call `vf.grade_submission(task, trace, runtime)`. This uses the
task's normal grader, verifier image, timeouts, and retries without taking ownership
of the supplied solver runtime.
