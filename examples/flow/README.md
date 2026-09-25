# Flow

A `Flow` subclass owns stages, shared services and scheduling policy. A `Job[Data]` is
one independently progressing instance, identified by a caller-chosen ID. It is passed
to a `@stage` method; authors do not subclass it.

```python
self.create(id="task-7", stage="author", data=TaskData(...))

# Apply a control immediately, from a stage or a monitor.
self.apply("campaign", Transition(stage="fix", status="ready"))

# Return from a stage to publish this job's next checkpoint.
return Transition(stage="evaluate", status="ready", data=job.data, outcome="authored")
```

`setup()` runs on every launch. `create()` preserves existing checkpoints, so use stable
IDs. Returning a `Transition` publishes only after the stage finishes successfully;
`apply()` publishes immediately, even when targeting the current job. It does not
interrupt that job's executing stage. Explicit controls override the returned transition
field by field; later explicit controls win over earlier ones. Omitted fields stay unchanged.
`reason` describes the state; `outcome` and `report` label the recorded transition.

Each job keeps one atomic `jobs/<id>/state.json` checkpoint. A short write lock protects
updates and increments its revision; the execution lock spans the stage. Mutating
`job.data` changes only the working copy: publish it with `Transition(data=job.data)`.
Data is a complete typed snapshot, never a patch. Immediate replacements require an
inactive job and `expected=job.state().revision`; stale revisions are rejected. Stage
returns need no explicit revision. Events live in `transitions.jsonl`.

Start with these short, runnable plugins:

- [single_agent.py](../../verifiers/v1/flows/single_agent.py): one agent per prompt.
- [parallel.py](../../verifiers/v1/flows/parallel.py): concurrent agents; retain successes if a sibling fails.
- [draft_review.py](../../verifiers/v1/flows/draft_review.py): two stages exchange a Git artifact revision.
- [partial_calls.py](../../verifiers/v1/flows/partial_calls.py): try partial failure and resume entirely offline.
- [cross_job.py](../../verifiers/v1/flows/cross_job.py): a repair stage steers a waiting task into review, entirely offline.

Export exactly one Flow subclass in the installed package's `__all__`. Its `Flow[Config]`
specialization selects its Pydantic config. Prime-RL handles launch paths and logging:

```sh
uv run flow --flow.id single-agent --flow.model MODEL --run.name answers
uv run flow --flow.id partial-calls --run.name demo --dashboard false
uv run flow inspect --root outputs/demo
uv run flow apply --root outputs/demo --job task --status ready
uv run flow --flow.id partial-calls --run.name demo --flow.available true --dashboard false
```

The offline example executes eight calls on its first launch, then only the two missing
results on the second. Its `available` flag describes an outage, so it is deliberately absent
from reuse inputs. Real pipelines declare their own meaningful task/model/grading inputs.

## Calls and files

`self.agents.solver.run(task, key=..., cache_inputs=...)` uses native agent execution, retries and
traces. Pass a key and a JSON object of cache dependencies together, or omit both to run every
time. Use `{}` for no dependencies. Successful results are reused from disk; failures remain
retryable. Agent `run()` returns a trace or raises; `attempt()`
returns success or failure. Host functions use `self.call(func, ..., output=ResultType)`
or `self.attempt(...)` with the same contract.

`self.gather(...)` waits for every child before propagating failures and settles children
on cancellation. Use `attempt()` to inspect individual outcomes, or `call()`/agent `run()`
when failure should raise. Successful keyed calls remain reusable either way.

Native Agent owns retries. Flow saves only the final trace (or the current partial trace on
cancellation), and token totals count only that trace. Live traces use the shared delta
format in `live/<trace_id>.jsonl`. A retry discards the previous live file; completion saves
the trace before removing its live file. Prime-RL's live viewer follows these updates.

For custom turn loops, pass an async `interact(interaction)` function to the agent's `run`
or `attempt`. It drives native `Interaction.turn`; the complete interaction is one recorded
call. To share a sandbox, use `async with solver.provision(task) as box`, then pass
`runtime=box` to agents that borrow it. The caller owns that box's lifetime.

Flow shares native interception, including elastic multiplexing. Tunnels are chosen at
launch from configured agent runtimes. Borrowed runtimes and task tool servers must be
reachable under that choice: an all-local configuration cannot introduce a remote consumer
of Verifiers' state channel later. External tool URLs do not use that channel. Flow does not
register state credentials for supplied shared tool servers; use task-scoped tools instead.

Reuse restores a value or trace, **never sandbox side effects**. `GitArtifacts(job.path)`
optionally preserves work products independently of workflow state; publish the
chosen revision in `Transition(data=job.data)`. External effects before returning a
transition are not rolled back by a hold.

## Agent control

A monitoring coding agent reads published job state, `transitions.jsonl`, call records and
traces (including `live/` deltas). `inspect` exposes current state and whether execution is active.
`apply` changes the next stage/status or appends a note; it does not interrupt a model's
conversation. Live controls survive stage publication. `job.before` and `job.notes` are
stage-start snapshots; `job.state()` reads current published state. Publish edits to the
stage's `job.data` explicitly with a transition. Operator data replacements require a settled job
and its inspected revision (`--data-file data.json --expected REVISION`).

`admit(job)` sees accepted executions in `self.active`, including their original executing
stage after a live route. Pipelines define barriers and success policy; `run()` returns job
states and an `idle` or `draining` reason. Use `flow.stay_alive = true` to wait for new work.
Stage concurrency is unbounded unless `flow.pools.jobs` is configured.

`flow drain --root outputs/demo` finishes running calls and stops new work. Remove the
`drain` file to launch again. Ctrl-C drains once and cancels on a second signal.
`pools.json` holds the current named limits; replace it atomically to resize them.
For a run configured with a `jobs` pool:

```sh
printf '%s\n' '{"jobs": 2}' > outputs/my-run/pools.json.tmp
mv outputs/my-run/pools.json.tmp outputs/my-run/pools.json
```

Flow checks it about every two seconds. Keep the same pool names and positive integer
limits. Lowering a limit lets accepted work finish. Invalid edits leave the previous limits
in effect and log the error; recovery decisions remain with the monitor.
