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

## Checkpoints

Filesystem checkpoints are a lifecycle operation, so they live on the backend:
`runtime.backend.checkpoint()` saves the box's files and returns an id once it is
restorable, `delete_checkpoint(id)` removes one (also from a backend that never
started), and `supports_checkpoint` says whether a backend has both (Prime does).
`restore_config(config, id)` is the runtime policy that starts a fresh box from a
checkpoint; such a backend reports `restored`. Files survive a checkpoint, processes
do not. A box `Agent.provision()` hands out has an owner (`vf.box_of(runtime)`, kept
on the backend so every view of the box shares it) that checkpoints it for the runs
borrowing it and restores it in place: `RuntimeBackend.adopt` moves a backend onto
a restored box, so every view bound to it follows. See
[Shared boxes](agent.md#shared-boxes).

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
provider details. Local Docker Compose requires
`agent.deployment.trust_compose = true` because task-authored Compose can access
host files and Docker privileges.

For model-free validation of an isolated task, finalize the submission and collect
its artifacts, then call `vf.grade_submission(task, trace, runtime)`. This uses the
task's normal grader, verifier image, timeouts, and retries without taking ownership
of the supplied solver runtime. A composed task needs its services, so provision its
deployment yourself (`backend.deploy(spec=task.deployment_spec(), ...)`), run the
task's `setup` and `finalize` in `deployment.workspace`, and pass the deployment:
`vf.grade_submission(task, trace, runtime, deployment=deployment)` stops the workspace,
collects sidecar evidence, and releases the services before the verifier starts, as a
rollout does. The caller still closes the deployment. `on_score=` runs a callback
after task scoring in the verifier, as `Agent.run(on_score=...)` does: a judge agent
borrowing the verifier's box, inside the scoring budget and retries, whose failure
commits none of the attempt's scores. `vf.check_restore(artifacts,
protected=...)` checks a collected submission on the host, before any grader exists,
for every defect `restore` would refuse.
