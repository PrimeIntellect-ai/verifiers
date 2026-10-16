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

**Service VMs** are task services on Prime VMs of their own. A Prime VM's network
policy covers the whole VM, Compose services included, so a service that needs other
egress, e.g. a tool server that calls public APIs while the workspace stays offline,
returns a `ServiceVMSpec` instead:

```python
def deployment_spec(self):
    return vf.ServiceVMSpec(
        {
            "tools": vf.ServiceVM(
                image="example/tools:1",
                command=["serve", "--port", "8000"],
                environment={"API_KEY": self.config.api_key},
                readiness=vf.Probe(["curl", "-sf", "localhost:8000/health"]),
                cpu=4,
                memory=8,
            ),
        }
    )
```

Each service keeps unrestricted egress. The workspace (`main`) is the agent's own VM,
under the task's policy. The VMs boot together, and a LAN network per rollout lets the
workspace's VM open connections to the services while nothing can open one to the
workspace. The workspace reaches a service at its LAN name,
`deployment.hosts[name]` (`<sandbox-id>.sandbox.internal`). A service's environment
reaches its VM only.

Task hooks can receive `deployment` and use `deployment.target(name)` to access a
named service through a runtime view. A task tool server can run in a named service
(`ToolsetConfig.service`: a Compose service on a Prime or Modal VM, or a service VM);
the harness reaches it over the Compose network by the service's name, or over the LAN,
so the server can front services the workspace cannot reach. `finalize_services()` collects sidecar evidence
before verification. See [Harbor's Compose support](harbor.md#docker-compose) for
provider details. Local Docker Compose requires
`agent.deployment.trust_compose = true` because task-authored Compose can access
host files and Docker privileges.

For model-free validation of an isolated task, finalize the submission and collect
its artifacts, then call `vf.grade_submission(task, trace, runtime)`. This uses the
task's normal grader, verifier image, timeouts, and retries without taking ownership
of the supplied solver runtime.
