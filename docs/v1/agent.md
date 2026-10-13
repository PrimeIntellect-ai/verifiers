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

## Shared boxes

`agent.provision(task)` hands out a box that runs borrow with `run(task, runtime=box)` or
`interaction(task, runtime=box)`; several seats can share it (a control agent and its
reviewer). The box comes with an owner, `vf.box_of(box)`, and the owner does what a
borrowing run may not:

- **Checkpoints.** A borrowed run checkpoints the box through the owner, one checkpoint at
  a time across all seats: per call when the run's `save = "per_call"` or the provisioning
  agent's is, and on `interaction.checkpoint()`. Each checkpoint is recorded on the run's
  own trace, matching that run's conversation.
- **Restore.** With `interaction_retries`, a turn that loses a borrowed box has the owner
  start a fresh box from the latest checkpoint any seat took and swap it in under the same
  runtime object, so every seat holding the box continues on the restored one. The turn
  needs a checkpoint of its own trace to continue from.

A box made any other way (`provision_runtime`) has no owner: runs on it are never
checkpointed or retried.
