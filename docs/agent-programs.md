---
title: "Agent Programs"
description: "Program over agents: one executable arrow, placement as a parameter, chaining as plain functions"
---

## The Agent

An `Agent` is a reusable value: a **harness** (a concrete `Harness` object — the program
that drives the model), a **model context** (`ModelContext`: model + client + sampling),
and a **runtime policy** (where a run's box comes from by default). It has one executable arrow:

```python
import verifiers.v1 as vf
from verifiers.v1.harnesses.default import DefaultHarness, DefaultHarnessConfig

ctx = vf.ModelContext(
    model="z-ai/glm-5.2", client=vf.resolve_client(vf.EvalClientConfig())
)
solver = vf.Agent(DefaultHarness(DefaultHarnessConfig()), ctx)
trace = await solver.run(vf.Task(idx=0, prompt="What is 2+2?"))
```

Construction is fully explicit — the harness is an object you build, and the client is
yours to build and **share**: agents on the same endpoint should share one `Client` (one
connection pool). prime-rl hands agents its renderer client through the same
`ModelContext`.

Every run is a standard `Rollout` — staged lifecycle, typed error attribution,
token-true trace capture — so anything a program produces is evaluable and trainable.
Everything beyond the arrow is a parameter, not a concept:

- **`taskset=`** attaches judgement (`@reward` / `@metric`, `setup` / `finalize`) to a
  run. Omitted, the run is unscored — a pure `Task -> Trace` arrow.
- **`runtime=`** places the run into a live box (borrowed — the run neither starts nor
  tears it down) instead of provisioning a fresh one. `agent.provision(task)` hands the
  program a box to place runs into.
- **`ctx=`** replaces the agent's model context per run
  (`dataclasses.replace(agent.ctx, model=...)` for a judge sweeping models).

Entered as an async context manager, an Agent owns an interception pool so concurrent
runs share servers and tunnels (`async with agent: ...`); un-entered, each run brings up
its own — fine for scripts.

Chaining needs no framework: mint the next `Task` from earlier traces with a plain
function, stamping `Task.sources` (trace ids) and `Task.relation` so lineage survives
into the traces.

## Same-box agentic judging

One box, two agents, sequential — the judge audits the solver's *world*, not a paste of
its transcript:

```python
sandbox = vf.PrimeConfig()
harness = DefaultHarness(DefaultHarnessConfig())
client = vf.resolve_client(vf.EvalClientConfig())
solver = vf.Agent(harness, vf.ModelContext(model="z-ai/glm-5.2", client=client), sandbox)
judge = vf.Agent(harness, vf.ModelContext(model="openai/gpt-5.4-mini", client=client), sandbox)

def judge_task(solver_trace: vf.Trace) -> vf.Task:
    return vf.Task(
        idx=0,
        prompt=(
            "Audit the agent whose trajectory is at /app/evidence/trace.json and whose "
            "work product is at /app/answer.txt. Recompute the expected result yourself "
            'and reply with ONLY {"verdict": "correct" | "incorrect", "reasoning": "..."}'
        ),
        sources=(solver_trace.id,),
        relation="judges",
    )

task = vf.Task(idx=0, prompt="Compute the sum of the first 100 primes into /app/answer.txt")

async with solver.provision(task) as box:
    solver_trace = await solver.run(task, runtime=box)
    await box.write("/app/evidence/trace.json",
                    json.dumps(solver_trace.to_record()).encode())
    verdict_trace = await judge.run(judge_task(solver_trace), runtime=box)
```

Both traces carry `info["agent"]` (harness, model, runtime type + descriptor, borrowed
flag), so "these two runs shared a box" stays a queryable relation afterwards.

## Proposer → solvers

Fan-out with judgement attached, written as a script:

```python
class ProposedTask(vf.Task):
    answer: str

class SolveTaskset(vf.Taskset[ProposedTask, vf.TasksetConfig]):
    @vf.reward
    async def correct(self, task: ProposedTask, trace: vf.Trace) -> float:
        ...  # compare the trace's final answer against task.answer

proposer_trace = await proposer.run(vf.Task(idx=0, prompt=PROPOSE))
task = mint_task(proposer_trace)          # your traces -> Task function
taskset = SolveTaskset(vf.TasksetConfig())
traces = await asyncio.gather(*(solver.run(task, taskset=taskset) for _ in range(8)))
```

Fan-out is plain `asyncio.gather` — each run gets its own fresh box, and the entered
agent's interception pool keeps N concurrent runs cheap. The Agent deliberately has no
group verb: each run scores its rollout on its own, and comparing siblings — relative
success, preference, advantages — belongs to whoever gathered the traces (in training,
prime-rl samples the group). The structure of a multi-agent program lives on the traces
themselves, via the lineage stamps.

Reward/metric handlers are `async def` — a sync handler fails at scoring time.

## Placement rules

- A **fresh box** per run is the default: the agent's runtime policy, resolved per task
  (`Task.image` / `workdir` / `resources`).
- A **borrowed box** (`runtime=`) is its creator's to tear down; runs in it skip
  provisioning entirely. Place any number of runs — by one agent or several — into one
  box; their traces never entangle (interception sessions are per-run). Borrowing a box
  its owner already tore down raises immediately — a lifetime bug in the program, not a
  rollout error on the trace.
- An adversarial caveat: an agent that runs in a box can tamper with evidence already in
  it. For hack detection, prefer writing the evidence *after* the suspect run (as above),
  or judge in a separate box.
