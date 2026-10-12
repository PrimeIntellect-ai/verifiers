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

## Runtime handles

A `Runtime` owns a provisioned resource: startup, network policy, and
teardown. A `RuntimeHandle` provides process and file access to that resource,
with immutable user, `HOME`, and environment bindings. `runtime.handle()` creates
an access handle without provisioning another box. `handle.with_user(...)` and
`handle.with_env(...)` return new handles; existing handles retain their bindings.
Working directory comes from the runtime config.

Task and harness hooks receive handles through their `runtime` argument.
`Agent.provision()` lends a handle whose owning context controls cleanup. The
handle has no start/stop methods; its `runtime` property identifies the owner.
Backend permissions enforce the selected user. A handle is not a security boundary.

Trusted task setup and grading use the default-user handle. Agent execution and
colocated tools use the task's agent-user handle, resolved after task setup.
Provider operations receive this handle explicitly, so concurrent users never
mutate the owner's identity.

## Verification placement

`agent.verifier.mode` selects `task` (the default), `shared`, or `isolated` scoring.
Tasks declare their default through `TaskData.verifier_mode`. Isolated verification
restores collected artifacts into a fresh runtime before task scoring.
`agent.verifier.runtime` and `agent.verifier.env` override the isolated verifier;
`agent.verifier.retries` controls additional fresh attempts.
