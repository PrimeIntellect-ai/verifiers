---
title: "Agent"
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
trace = await solver.run(vf.Task(vf.TaskData(idx=0, prompt="What is 2+2?")))
```

Construction is fully explicit — the harness is an object you build, and the client is
yours to build and **share**: agents on the same endpoint should share one `Client` (one
connection pool). prime-rl hands agents its renderer client through the same
`ModelContext`.

Every run is a standard `Rollout` — staged lifecycle, typed error attribution,
token-true trace capture — so anything a program produces is evaluable and trainable.
Everything beyond the arrow is a parameter, not a concept:

- **judgement rides on the task.** A `Task` subclass's hooks (`setup` / `finalize`) and
  signals (`@reward` / `@metric`) run exactly as in an eval; a plain base
  `vf.Task(data)` has neither, so the run is unscored — a pure `Task -> Trace` arrow.
  Impossible pairings are refused up front, exactly as in an eval: a task whose tools or
  user simulator the harness doesn't drive, or a `NEEDS_CONTAINER` task on the
  subprocess runtime.
- **`runtime=`** places the run into a live box (borrowed — the run neither starts nor
  tears it down) instead of provisioning a fresh one. `agent.provision(task)` hands the
  program a box to place runs into. A different model is a different agent — construct
  another `Agent` (sharing the client, and the interception pool via `interception=`)
  rather than swapping contexts per run.

Interception follows the same borrowing story as runtimes: pass a live, already-entered
`Interception` at construction (`vf.Agent(..., interception=pool)`) and several agents
share one pool of servers and tunnels — its owner keeps the lifecycle, the agent only
acquires slots. Without one, an Agent entered as an async context manager owns an
elastic pool so concurrent runs share servers (`async with agent: ...`); un-entered,
each run brings up its own per-rollout server — fine for scripts.

Chaining needs no framework: mint the next task's `TaskData` from earlier traces with a
plain function, stamping `sources` (trace ids) and `relation` so lineage survives into
the traces.

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
    return vf.Task(vf.TaskData(
        idx=0,
        prompt=(
            "Audit the agent whose trajectory is at /app/evidence/trace.json and whose "
            "work product is at /app/answer.txt. Recompute the expected result yourself "
            'and reply with ONLY {"verdict": "correct" | "incorrect", "reasoning": "..."}'
        ),
        sources=(solver_trace.id,),
        relation="judges",
    ))

task = vf.Task(vf.TaskData(
    idx=0, prompt="Compute the sum of the first 100 primes into /app/answer.txt"
))

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
class ProposedData(vf.TaskData):
    answer: str

class ProposedTask(vf.Task[ProposedData]):
    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        ...  # compare the trace's final answer against self.data.answer

proposer_trace = await proposer.run(vf.Task(vf.TaskData(idx=0, prompt=PROPOSE)))
task = mint_task(proposer_trace)          # your traces -> ProposedTask function
traces = await asyncio.gather(*(solver.run(task) for _ in range(8)))
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
  (`TaskData.image` / `workdir` / `resources`).
- A **borrowed box** (`runtime=`) is its creator's to tear down; runs in it skip
  provisioning entirely. Place any number of runs — by one agent or several — into one
  box; their traces never entangle (interception sessions are per-run). Borrowing a box
  its owner already tore down raises immediately, and a box torn down mid-run raises at
  the awaited `run()` (the raw failure chained underneath) — lifetime bugs in the
  program, not rollout errors on the trace.
- A borrowed box is never **re-provisioned**, so a task's placement fields can't be
  honored there. Where the provisioning path refuses, borrowing refuses the same: a
  task `image` on a subprocess-backed box raises. A container box running a different
  image only logs a warning — running in an existing world is the point of borrowing.
- An adversarial caveat: an agent that runs in a box can tamper with evidence already in
  it. For hack detection, prefer writing the evidence *after* the suspect run (as above),
  or judge in a separate box.
