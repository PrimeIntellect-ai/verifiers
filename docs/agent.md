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

See [Runtime](runtime.md) for execution handles, resource ownership, and deployment
of the harness, task workspace, and supporting services.

## Verification placement

`agent.scoring.mode` selects `shared` or `isolated` scoring. Tasks declare defaults
through `TaskData.scoring`, using the same `ScoringConfig`. Explicit agent fields
override task fields; otherwise scoring defaults to `shared`. Isolated verification
restores collected artifacts into a fresh runtime before task scoring.
`agent.scoring.runtime` overrides the isolated verifier's runtime.
`agent.scoring.env` merges over task scoring variables in either placement;
`${VAR}` and `${VAR:-default}` templates resolve only when scoring starts.
`agent.scoring.retries` uses `RetryConfig` for isolated scoring, with a default
`max_retries = 2`. Its rules select retryable errors; retries reuse the completed
submission in a fresh scoring environment. Shared scoring is not retried.
