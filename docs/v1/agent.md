# Agent

An `Agent` runs a model and harness on a task and returns a trace. Use `AgentConfig` to choose its settings:

```python
import verifiers.v1 as vf


async with vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2")) as solver:
    trace = await solver.run(vf.Task(vf.TaskData(prompt="What is 2+2?")))
```

Use `agent.interaction(task)` to send follow-up messages. Each `turn()` runs the
agent until it finishes or needs another user message. It may make several model
and tool calls along the way. If the task has a prompt, start with `turn()`.
If `prompt=None`, supply the first message with `turn(message)`:

```python
async with agent.interaction(task) as interaction:
    # This example assumes task.data.prompt is None.
    segment = await interaction.turn("hello")
    if not segment.terminated:
        segment = await interaction.turn(f"you said: {segment.last_reply}")

trace = interaction.trace
```
