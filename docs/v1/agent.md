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

`agent.with_config(**update)` is a copy of the agent with config fields replaced, for
what one run needs that its config cannot say ahead. Inside an env, the copy of an
episode's agent is still that episode's agent: its traces join the episode.

```python
async def run(self, task, agents):
    harness = agents.solver.config.harness.model_copy(
        update={"mcp_header_env": {"world": {"Authorization": per_run_variable}}}
    )
    await agents.solver.with_config(harness=harness).run(task)
```
