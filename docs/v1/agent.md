# The Agent

An `Agent` is a configured **harness** (the program that drives the model), a **model**, and a **runtime** (where the harness executes), built from an `AgentConfig` alone. An agent is given a `Task` and produces a `Trace`.

```python
import verifiers.v1 as vf

async with vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2")) as solver:
    trace = await solver.run(vf.Task(vf.TaskData(prompt="What is 2+2?")))
```

Every run is a standard rollout producing a `vf.Trace`. By default, the agent is self-contained: its context owns the model client and shared interception server, while each run owns its runtime and any per-run interception machinery.

Exiting the context closes an agent-owned client, so create a new agent for later runs; injected clients remain caller-owned.

## Interactions

`agent.interaction(task)` holds a rollout open turn by turn. The caller acts as the
user, and each `turn()` runs one harness segment before returning a `vf.Segment`.

```python
async with agent.interaction(task) as interaction:
    segment = await interaction.turn("hello")
    if not segment.terminated:
        segment = await interaction.turn(f"you said: {segment.last_reply}")

trace = interaction.trace
```

Each `Segment.messages` contains the assistant messages, tool calls, and tool
results produced by that harness segment. `Segment.last_reply` is shorthand for
the final assistant message's text.

A prompted task speaks first through a bare `turn()`; a prompt-less task starts
with `turn(message)`. Leaving the context closes the exchange as `user_closed`
and finishes scoring. `interaction(mask_prompt=True)` keeps a scenario prompt
available to the task while hiding it from the assistant.

## Borrowed Resources

At scale (large evals, training), per-run machinery adds up. `make_agent` accepts live resources to borrow instead of creating its own.

### Interception Server

Pass `interception=` to reuse interception servers.

```python
from verifiers.v1.interception import InterceptionServer

async with InterceptionServer() as server:
    solver = vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2"), interception=server)
    judge = vf.make_agent(vf.AgentConfig(model="openai/gpt-5.4-mini"), interception=server)
    ...
```

### Client

Pass `client=` to reuse an existing client — agents on the same endpoint should share a single `Client` (one connection pool).

```python
client = vf.resolve_client(vf.EvalClientConfig())

solver = vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2"), client=client)
judge = vf.make_agent(vf.AgentConfig(model="openai/gpt-5.4-mini"), client=client)
```

The caller is responsible for correctly handling the lifecycle of such borrowed resources: they must be live for every run placed on them, and the agent never tears them down.

## Trace

A `Trace` holds all information on a single agent's rollout: the message graph, model calls, usage, timing, the rewards, metrics, and errors it recorded. Whatever a run did, the trace is the artifact you store, chain, and train on.

## Examples

### Chaining agents

Agents are chained via `vf.Task`. For example, a common pattern is one agent's task depending on another agent's task. Use the `Task.from_trace` API, to construct such "glue" tasks.

```python
class ProposedData(vf.TaskData):
    answer: str

class ProposedTask(vf.Task[ProposedData]):
    @classmethod
    def from_trace(cls, proposed: vf.Trace) -> "ProposedTask":
        ...  # parse the proposer's contract into ProposedData

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        ...  # compare the trace's final answer against self.data.answer

proposed = await proposer.run(vf.Task(vf.TaskData(prompt=PROPOSE)))
task = ProposedTask.from_trace(proposed)
async with asyncio.TaskGroup() as tg:
    for _ in range(8):
        tg.create_task(solver.run(task))
```

### Shared Runtimes

Runtimes can be borrowed too: `agent.provision(task)` provisions a box from the agent's runtime policy as a context manager, and `run(..., runtime=box)` places a run into it instead of provisioning a fresh one. Chained agents then share one file system — here, the judge inspects what the solver left behind:

```python
task = vf.Task(vf.TaskData(prompt="Sum the first 100 primes into answer.txt"))
audit = vf.Task(vf.TaskData(prompt="Recompute the sum and verify answer.txt"))

async with (
    vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2")) as solver,
    vf.make_agent(vf.AgentConfig(model="openai/gpt-5.4-mini")) as judge,
):
    async with solver.provision(task) as box:
        solution = await solver.run(task, runtime=box)
        verdict = await judge.run(audit, runtime=box)
```

The box lives exactly as long as the `async with`: borrowed runs never
provision or tear it down.
