# The Agent

An `Agent` is a configured **harness** (the program that drives the model), a
**model**, and a **runtime policy** (where a run's box comes from), built from its
`AgentConfig` alone and runnable on any task:

```python
import verifiers.v1 as vf

solver = vf.make_agent(vf.AgentConfig(model="z-ai/glm-5.2"))
trace = await solver.run(vf.Task(vf.TaskData(prompt="What is 2+2?")))
```

The config carries everything declarative: the harness (`None` = the built-in
`bash`), the model, an endpoint override, sampling, per-stage timeouts, turn/token
caps, and whole-run `retries`. Live resources are **injected and borrowed, never
configured**: `make_agent(config, client=...)` shares one connection pool across
agents on the same endpoint (prime-rl hands agents its renderer client the same
way), and `interception=` shares a live interception pool — its owner keeps the
lifecycle. Without one, an entered agent (`async with agent:`) owns a single
interception server multiplexing its concurrent runs; un-entered, each run brings
up its own per-rollout server — fine for scripts.

Every run is a standard rollout — staged lifecycle, typed error attribution,
token-true trace capture — so anything a program produces is evaluable and
trainable. Everything beyond the arrow is a parameter, not a concept:

- **judgement rides on the task.** A `Task` subclass's hooks (`setup` / `finalize`)
  and signals (`@reward` / `@metric`) run exactly as in an eval; a plain base
  `vf.Task(data)` has neither, so the run is unscored — a pure `Task -> Trace`
  arrow. Impossible pairings are refused per run, on the task the agent actually
  receives: a task whose tools the harness doesn't drive, or a `NEEDS_CONTAINER`
  task on the subprocess runtime.
- **`runtime=`** places the run into a live box (borrowed — the run neither starts
  nor tears it down) instead of provisioning a fresh one from the harness config's
  runtime. `agent.provision(task)` hands the program a box to place runs into. A
  different model is a different agent — construct another `Agent` (injecting the
  shared client) rather than swapping contexts per run.
- **`shared_tools=`** wires live `SharedToolServer`s (taskset-scoped MCP, served
  once by their owner) into that run. Per run, not per agent, and borrowed like the
  others — counted in the pairing check.
- **retries are the agent's own**: `run` reruns its rollout while the trace ends
  with a retryable error (`--env.<agent>.retries.max_retries`, flat
  include/exclude), with backoff — never into a borrowed box, whose state is no
  longer the task's start state.

Chaining is a `Task` classmethod: mint the next task from earlier traces with
`YourTask.from_trace(...)` and hand it to the next agent.

## Same-box agentic judging

One box, two agents, sequential — the judge audits the solver's *world*, not a
paste of its transcript (the bundled `agentic-judge` env is the packaged version
of this shape):

```python
client = vf.resolve_client(vf.EvalClientConfig())  # one endpoint, one shared pool
solver = vf.make_agent(
    vf.AgentConfig(harness=vf.BashHarnessConfig(runtime=vf.PrimeConfig()),
                   model="z-ai/glm-5.2"),
    client=client,
)
judge = vf.make_agent(
    vf.AgentConfig(harness=vf.BashHarnessConfig(runtime=vf.PrimeConfig()),
                   model="openai/gpt-5.4-mini"),
    client=client,
)

class AuditTask(vf.Task):
    @classmethod
    def from_trace(cls, solution: vf.Trace) -> "AuditTask":
        return cls(vf.TaskData(
            idx=solution.task.data.idx,
            prompt=(
                "Audit the agent whose trajectory is at /app/evidence/trace.json "
                "and whose work product is at /app/answer.txt. Recompute the "
                "expected result yourself and reply with ONLY "
                '{"verdict": "correct" | "incorrect", "reasoning": "..."}'
            ),
        ))

task = vf.Task(vf.TaskData(
    idx=0, prompt="Compute the sum of the first 100 primes into /app/answer.txt"
))

async with solver.provision(task) as box:
    solution = await solver.run(task, runtime=box)
    await box.write("/app/evidence/trace.json",
                    json.dumps(solution.to_record()).encode())
    verdict = await judge.run(AuditTask.from_trace(solution), runtime=box)
```

Every trace carries its agent identity typed (`trace.agent` — model, sampling,
harness config, name, trainability) and its placement (`trace.runtime`, including
`borrowed`), so "these two runs shared a box" stays a queryable relation
afterwards.

## Proposer → solvers

Fan-out with judgement attached, written as a script:

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

proposed = await proposer.run(vf.Task(vf.TaskData(idx=0, prompt=PROPOSE)))
task = ProposedTask.from_trace(proposed)
async with asyncio.TaskGroup() as tg:
    for _ in range(8):
        tg.create_task(solver.run(task))
```

Each run gets its own fresh box, and the entered agent's interception server
multiplexes the N concurrent runs. The Agent deliberately has no group verb: each
run scores its rollout on its own, and comparing siblings — relative success,
preference, advantages — belongs to whoever gathered the traces (in training,
prime-rl samples the group; in an `Environment`, `finalize()` compares the finished
traces).

Reward/metric handlers are `async def` — a sync handler is refused at decoration.

## Placement rules

- A **fresh box** per run is the default: the harness config's runtime policy,
  resolved per task (`TaskData.image` / `workdir` / `resources`).
- A **borrowed box** (`runtime=`) is its creator's to tear down; runs in it skip
  provisioning entirely. Place any number of runs — by one agent or several — into
  one box; their traces never entangle (interception sessions are per-run).
  Borrowing a box its owner already tore down raises immediately, and a box torn
  down mid-run raises at the awaited `run()` (the raw failure chained underneath) —
  lifetime bugs in the program, not rollout errors on the trace.
- A borrowed box is never **re-provisioned**, so a task's placement fields can't be
  honored there. Where the provisioning path refuses, borrowing refuses the same: a
  task `image` on a subprocess-backed box raises. A container box running a
  different image only logs a warning — running in an existing world is the point
  of borrowing.
- An adversarial caveat: an agent that runs in a box can tamper with evidence
  already in it. For hack detection, prefer writing the evidence *after* the
  suspect run (as above), or judge in a separate box.
