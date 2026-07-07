---
title: "Agent Programs"
description: "Program over agents: one executable arrow, placement as a parameter, chaining as plain functions"
---

## The Agent

An `Agent` is a reusable value: a **harness** (the program that drives the model), a
**rollout context** (client + model + sampling), and a **runtime policy** (where a run's
box comes from by default). It has one executable arrow:

```python
import verifiers.v1 as vf

solver = vf.Agent("default", vf.make_context("z-ai/glm-5.2"))
trace = await solver.run(vf.Task(idx=0, prompt="What is 2+2?"))
```

Every run is a standard `Rollout` — staged lifecycle, typed error attribution,
token-true trace capture — so anything a program produces is evaluable and trainable.
Everything beyond the arrow is a parameter, not a concept:

- **`taskset=`** attaches judgement (`@reward` / `@metric`, `setup` / `finalize`) to a
  run. Omitted, the run is unscored — a pure `Task -> Trace` arrow.
- **`runtime=`** places the run into a live box (borrowed — the run neither starts nor
  tears it down) instead of provisioning a fresh one. `agent.provision(task)` hands the
  program a box to place runs into.
- **`model=` / `sampling=`** override the agent's context per run.

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
solver = vf.Agent("bash", vf.make_context("z-ai/glm-5.2"), sandbox)
judge = vf.Agent("bash", vf.make_context("openai/gpt-5.4-mini"), sandbox)

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

async with solver.provision(task) as box:
    solver_trace = await solver.run(task, runtime=box)
    await box.write("/app/evidence/trace.json",
                    json.dumps(solver_trace.to_record()).encode())
    verdict_trace = await judge.run(judge_task(solver_trace), runtime=box)
```

Both traces carry `info["agent"]` (harness, model, runtime type + descriptor, borrowed
flag), so "these two runs shared a box" stays a queryable relation afterwards.

## Proposer → solvers

Fan-out with judgement attached — the shape a `ProposerSolverEnv` returns as a typed
topology, written as a script:

```python
class ProposedTask(vf.Task):
    answer: str

class SolveTaskset(vf.Taskset[ProposedTask, vf.TasksetConfig]):
    @vf.reward
    async def correct(self, task: ProposedTask, trace: vf.Trace) -> float:
        ...  # compare the trace's final answer against task.answer

proposer_trace = await proposer.run(vf.Task(idx=0, prompt=PROPOSE))
task = mint_task(proposer_trace)          # your traces -> Task function
traces = await solver.run_group(task, n=8, taskset=SolveTaskset(vf.TasksetConfig()))
```

`run_group` runs the task `n` times concurrently (each in its own fresh box) and, when
the taskset defines `@group_reward`s, scores them across the group — the same semantics
as an eval `Episode`, without its retry layer.

Reward/metric handlers are `async def` — a sync handler fails at scoring time.

## Placement rules

- A **fresh box** per run is the default: the agent's runtime policy, resolved per task
  (`Task.image` / `workdir` / `resources`).
- A **borrowed box** (`runtime=`) is its creator's to tear down; runs in it skip
  provisioning entirely. Place any number of runs — by one agent or several — into one
  box; their traces never entangle (interception sessions are per-run).
- An adversarial caveat: an agent that runs in a box can tamper with evidence already in
  it. For hack detection, prefer writing the evidence *after* the suspect run (as above),
  or judge in a separate box.
