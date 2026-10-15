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
| Task image, container-only harnesses, isolated scoring | `supports_image` | docker, prime, modal, e2b |
| Task network policy | config inherits `NetworkPolicyConfig` | prime, modal, e2b |
| Task `agent_user` | `supports_user` | docker, prime |
| Tool server in its own runtime | `supports_expose` | subprocess, docker, modal, e2b |
| Docker Compose services | `supports_compose` | prime, modal |

## Runtime plugins

Select `type = "custom"` to use a `RuntimeBackend` subclass from your own package:

```toml
[env.agent.runtime]
type = "custom"
import_path = "my_package.runtime.MyBackend"
region = "us-east-1"
```

The backend sets `config_class` to its own `CustomRuntimeConfig` subclass. That class
validates every field next to `import_path` when the config loads. The framework
constructs the backend as `MyBackend(config, name)` with the validated config, like
the built-in backends. A backend with `supports_image` declares `image` and `workdir`
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
    supports_image: ClassVar[bool] = True

    def __init__(self, config: MyConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = CustomRuntimeInfo(**config.model_dump())
```

A plugin implements `start`, `_run`, `_read`, `_write`, and `cleanup` (or `teardown`),
and optionally `_open_process` and `_run_background`. Install its package in every
process that loads the config, because validation imports the backend. The capability
attributes default to off. `tests/unit/runtimes/test_custom.py` has a complete
example.

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
provider details. Compose runs on Prime and Modal VMs.

For model-free validation of an isolated task, finalize the submission and collect
its artifacts, then call `vf.grade_submission(task, trace, runtime)`. This uses the
task's normal grader, verifier image, timeouts, and retries without taking ownership
of the supplied solver runtime.
