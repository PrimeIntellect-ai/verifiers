# Agent

An `Agent` runs a model and harness on a task and returns a trace. Use `AgentConfig` to choose its settings:

```python
import verifiers.v1 as vf


async with vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2")) as solver:
    trace = await solver.run(vf.Task(vf.TaskData(prompt="What is 2+2?")))
```

Use `agent.interaction(task)` to send follow-up messages. Each `turn()` runs one
harness segment, which may include several model and tool calls. Keep sending
follow-up messages until `segment.terminated`, or leave the context to end the
exchange yourself. After the agent's final reply, the next call returns an empty
terminated segment without consuming the supplied message. Closing the context
finishes the rollout, including scoring.

If the task has a prompt, start with `turn()`. If `prompt=None`, supply the first
message with `turn(message)`. This example stops after three model turns:

```python
import verifiers.v1 as vf


task = vf.Task(vf.TaskData(prompt=None))
config = vf.AgentConfig(model="z-ai/glm-5.2", max_turns=3)
async with vf.make_agent(config) as solver:
    async with solver.interaction(task) as interaction:
        segment = await interaction.turn("hello")
        while not segment.terminated:
            segment = await interaction.turn(f"you said: {segment.last_reply}")

trace = interaction.trace
```
